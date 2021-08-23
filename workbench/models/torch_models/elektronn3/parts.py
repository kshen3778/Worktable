# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Author: Martin Drawitsch

"""
This is a modified version of the U-Net CNN architecture for biomedical
image segmentation. U-Net was originally published in
https://arxiv.org/abs/1505.04597 by Ronneberger et al.
A pure-3D variant of U-Net has been proposed by Çiçek et al.
in https://arxiv.org/abs/1606.06650, but the below implementation
is based on the original U-Net paper, with several improvements.
This code is based on https://github.com/jaxony/unet-pytorch
(c) 2017 Jackson Huang, released under MIT License,
which implements (2D) U-Net with user-defined network depth
and a few other improvements of the original architecture.
Major differences of this version from Huang's code:
- Operates on 3D image data (5D tensors) instead of 2D data
- Uses 3D convolution, 3D pooling etc. by default
- planar_blocks architecture parameter for mixed 2D/3D convnets
  (see UNet class docstring for details)
- Improved tests (see the bottom of the file)
- Cleaned up parameter/variable names and formatting, changed default params
- Updated for PyTorch 1.3 and Python 3.6 (earlier versions unsupported)
- (Optional DEBUG mode for optional printing of debug information)
- Extended documentation
"""

import copy
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F


def get_conv(dim=3):
    """Chooses an implementation for a convolution layer."""
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d
    else:
        raise ValueError("dim has to be 2 or 3")


def get_convtranspose(dim=3):
    """Chooses an implementation for a transposed convolution layer."""
    if dim == 3:
        return nn.ConvTranspose3d
    elif dim == 2:
        return nn.ConvTranspose2d
    else:
        raise ValueError("dim has to be 2 or 3")


def get_maxpool(dim=3):
    """Chooses an implementation for a max-pooling layer."""
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d
    else:
        raise ValueError("dim has to be 2 or 3")


def get_normalization(normtype: str, num_channels: int, dim: int = 3):
    """Chooses an implementation for a batch normalization layer."""
    if normtype is None or normtype == "none":
        return nn.Identity()
    elif normtype.startswith("group"):
        if normtype == "group":
            num_groups = 8
        elif len(normtype) > len("group") and normtype[len("group") :].isdigit():
            num_groups = int(normtype[len("group") :])
        else:
            raise ValueError(
                f'normtype "{normtype}" not understood. It should be "group<G>",'
                f" where <G> is the number of groups."
            )
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif normtype == "instance":
        if dim == 3:
            return nn.InstanceNorm3d(num_channels)
        elif dim == 2:
            return nn.InstanceNorm2d(num_channels)
        else:
            raise ValueError("dim has to be 2 or 3")
    elif normtype == "batch":
        if dim == 3:
            return nn.BatchNorm3d(num_channels)
        elif dim == 2:
            return nn.BatchNorm2d(num_channels)
        else:
            raise ValueError("dim has to be 2 or 3")
    else:
        raise ValueError(
            f'Unknown normalization type "{normtype}".\n'
            'Valid choices are "batch", "instance", "group" or "group<G>",'
            "where <G> is the number of groups."
        )


def planar_kernel(x):
    """Returns a "planar" kernel shape (e.g. for 2D convolution in 3D space)
    that doesn't consider the first spatial dim (D)."""
    if isinstance(x, int):
        return (1, x, x)
    else:
        return x


def planar_pad(x):
    """Returns a "planar" padding shape that doesn't pad along the first spatial dim (D)."""
    if isinstance(x, int):
        return (0, x, x)
    else:
        return x


def conv3(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    planar=False,
    dim=3,
):
    """Returns an appropriate spatial convolution layer, depending on args.
    - dim=2: Conv2d with 3x3 kernel
    - dim=3 and planar=False: Conv3d with 3x3x3 kernel
    - dim=3 and planar=True: Conv3d with 1x3x3 kernel
    """
    if planar:
        stride = planar_kernel(stride)
        padding = planar_pad(padding)
        kernel_size = planar_kernel(kernel_size)
    return get_conv(dim)(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )


def upconv2(in_channels, out_channels, mode="transpose", planar=False, dim=3):
    """Returns a learned upsampling operator depending on args."""
    kernel_size = 2
    stride = 2
    if planar:
        kernel_size = planar_kernel(kernel_size)
        stride = planar_kernel(stride)
    if mode == "transpose":
        return get_convtranspose(dim)(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
    elif "resizeconv" in mode:
        if "linear" in mode:
            upsampling_mode = "trilinear" if dim == 3 else "bilinear"
        else:
            upsampling_mode = "nearest"
        rc_kernel_size = 1 if mode.endswith("1") else 3
        return ResizeConv(
            in_channels,
            out_channels,
            planar=planar,
            dim=dim,
            upsampling_mode=upsampling_mode,
            kernel_size=rc_kernel_size,
        )


def conv1(in_channels, out_channels, dim=3):
    """Returns a 1x1 or 1x1x1 convolution, depending on dim"""
    return get_conv(dim)(in_channels, out_channels, kernel_size=1)


def get_activation(activation):
    if isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leaky":
            return nn.LeakyReLU(negative_slope=0.1)
        elif activation == "prelu":
            return nn.PReLU(num_parameters=1)
        elif activation == "rrelu":
            return nn.RReLU()
        elif activation == "lin":
            return nn.Identity()
    else:
        # Deep copy is necessary in case of paremtrized activations
        return copy.deepcopy(activation)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        pooling=True,
        planar=False,
        activation="relu",
        normalization=None,
        full_norm=True,
        dim=3,
        conv_mode="same",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        self.dim = dim
        padding = 1 if "same" in conv_mode else 0

        self.conv1 = conv3(
            self.in_channels, self.out_channels, planar=planar, dim=dim, padding=padding
        )
        self.conv2 = conv3(
            self.out_channels,
            self.out_channels,
            planar=planar,
            dim=dim,
            padding=padding,
        )

        if self.pooling:
            kernel_size = 2
            if planar:
                kernel_size = planar_kernel(kernel_size)
            self.pool = get_maxpool(dim)(kernel_size=kernel_size)
            self.pool_ks = kernel_size
        else:
            self.pool = nn.Identity()
            self.pool_ks = (
                -123
            )  # Bogus value, will never be read. Only to satisfy TorchScript's static type system

        self.act1 = get_activation(activation)
        self.act2 = get_activation(activation)

        if full_norm:
            self.norm0 = get_normalization(normalization, self.out_channels, dim=dim)
        else:
            self.norm0 = nn.Identity()
        self.norm1 = get_normalization(normalization, self.out_channels, dim=dim)

    def check_poolable(self, y: torch.Tensor) -> None:
        """Before pooling, we manually check if the tensor is divisible by the pooling kernel
        size, because PyTorch doesn't throw an error if it's not divisible, but calculates
        the output shape by floor division instead. While this may make sense for other
        architectures, in U-Net this would lead to incorrect output shapes after upsampling.
        """
        # The code below looks stupidly repetitive, but currently it has to be this way to
        #  ensure source-level TorchScript compatibility. This is due to the static type system of
        #  TorchScript (ks can be None, int, Tuple[int, int] or Tuple[int, int, int]).
        #  TorchScript Exceptions don't support messages, so print statements are used for errors.
        ks = self.pool_ks
        if ks is None:
            return
        if isinstance(ks, int):  # given as scalar -> extend to spatial shape
            if self.dim == 3:
                for i in range(self.dim):
                    if y.shape[2 + i] % ks != 0:
                        print(f"\nCan't pool {y.shape[2:]} input by {ks}.\n")
                        raise Exception
            else:
                for i in range(self.dim):
                    if y.shape[2 + i] % ks != 0:
                        print(f"\nCan't pool {y.shape[2:]} input by {ks}.\n")
                        raise Exception
        else:
            for i in range(self.dim):
                if y.shape[2 + i] % ks[i] != 0:
                    print(f"\nCan't pool {y.shape[2:]} input by {ks}.\n")
                    raise Exception

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm0(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.norm1(y)
        y = self.act2(y)
        before_pool = y
        if self.pooling:
            self.check_poolable(y)
        y = self.pool(y)
        return y, before_pool


@torch.jit.script
def autocrop(
    from_down: torch.Tensor, from_up: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if from_down.shape[2:] != from_up.shape[2:]:
        # If VALID convolutions are used (not SAME), we need to center-crop to
        #  make features combinable.
        ds = from_down.shape[2:]
        us = from_up.shape[2:]
        assert ds[0] >= us[0]
        assert ds[1] >= us[1]
        if from_down.dim() == 4:
            from_down = from_down[
                :,
                :,
                ((ds[0] - us[0]) // 2) : ((ds[0] + us[0]) // 2),
                ((ds[1] - us[1]) // 2) : ((ds[1] + us[1]) // 2),
            ]
        elif from_down.dim() == 5:
            assert ds[2] >= us[2]
            from_down = from_down[
                :,
                :,
                ((ds[0] - us[0]) // 2) : ((ds[0] + us[0]) // 2),
                ((ds[1] - us[1]) // 2) : ((ds[1] + us[1]) // 2),
                ((ds[2] - us[2]) // 2) : ((ds[2] + us[2]) // 2),
            ]
    return from_down, from_up


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        merge_mode="concat",
        up_mode="transpose",
        planar=False,
        activation="relu",
        normalization=None,
        full_norm=True,
        dim=3,
        conv_mode="same",
        attention=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.normalization = normalization
        padding = 1 if "same" in conv_mode else 0

        self.upconv = upconv2(
            self.in_channels,
            self.out_channels,
            mode=self.up_mode,
            planar=planar,
            dim=dim,
        )

        if self.merge_mode == "concat":
            self.conv1 = conv3(
                2 * self.out_channels,
                self.out_channels,
                planar=planar,
                dim=dim,
                padding=padding,
            )
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3(
                self.out_channels,
                self.out_channels,
                planar=planar,
                dim=dim,
                padding=padding,
            )
        self.conv2 = conv3(
            self.out_channels,
            self.out_channels,
            planar=planar,
            dim=dim,
            padding=padding,
        )

        self.act0 = get_activation(activation)
        self.act1 = get_activation(activation)
        self.act2 = get_activation(activation)

        if full_norm:
            self.norm0 = get_normalization(normalization, self.out_channels, dim=dim)
            self.norm1 = get_normalization(normalization, self.out_channels, dim=dim)
        else:
            self.norm0 = nn.Identity()
            self.norm1 = nn.Identity()
        self.norm2 = get_normalization(normalization, self.out_channels, dim=dim)
        if attention:
            self.attention = GridAttention(
                in_channels=in_channels // 2, gating_channels=in_channels, dim=dim
            )
        else:
            self.attention = DummyAttention()
        self.att = None  # Field to store attention mask for later analysis

    def forward(self, enc, dec):
        """ Forward pass
        Arguments:
            enc: Tensor from the encoder pathway
            dec: Tensor from the decoder pathway (to be upconv'd)
        """

        updec = self.upconv(dec)
        enc, updec = autocrop(enc, updec)
        genc, att = self.attention(enc, dec)
        self.att = att
        updec = self.norm0(updec)
        updec = self.act0(updec)
        if self.merge_mode == "concat":
            mrg = torch.cat((updec, genc), 1)
        else:
            mrg = updec + genc
        y = self.conv1(mrg)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.act2(y)
        return y


class ResizeConv(nn.Module):
    """Upsamples by 2x and applies a convolution.
    This is meant as a replacement for transposed convolution to avoid
    checkerboard artifacts. See
    - https://distill.pub/2016/deconv-checkerboard/
    - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        planar=False,
        dim=3,
        upsampling_mode="nearest",
    ):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.scale_factor = 2
        if dim == 3 and planar:  # Only interpolate (H, W) dims, leave D as is
            self.scale_factor = planar_kernel(self.scale_factor)
        self.dim = dim
        self.upsample = nn.Upsample(
            scale_factor=self.scale_factor, mode=self.upsampling_mode
        )
        # TODO: Investigate if 3x3 or 1x1 conv makes more sense here and choose default accordingly
        # Preliminary notes:
        # - conv3 increases global parameter count by ~10%, compared to conv1 and is slower overall
        # - conv1 is the simplest way of aligning feature dimensions
        # - conv1 may be enough because in all common models later layers will apply conv3
        #   eventually, which could learn to perform the same task...
        #   But not exactly the same thing, because this layer operates on
        #   higher-dimensional features, which subsequent layers can't access
        #   (at least in U-Net out_channels == in_channels // 2).
        # --> Needs empirical evaluation
        if kernel_size == 3:
            self.conv = conv3(
                in_channels, out_channels, padding=1, planar=planar, dim=dim
            )
        elif kernel_size == 1:
            self.conv = conv1(in_channels, out_channels, dim=dim)
        else:
            raise ValueError(
                f"kernel_size={kernel_size} is not supported. Choose 1 or 3."
            )

    def forward(self, x):
        return self.conv(self.upsample(x))


class GridAttention(nn.Module):

    """Based on https://github.com/ozan-oktay/Attention-Gated-Networks
    Published in https://arxiv.org/abs/1804.03999
    if attention:
        self.attention = GridAttention(
            in_channels=in_channels // 2, gating_channels=in_channels, dim=dim
        )
    Note: Usually called in ublock...
    """

    def __init__(
        self,
        in_channels,
        gating_channels,
        inter_channels=None,
        dim=3,
        sub_sample_factor=2,
    ):
        super().__init__()

        assert dim in [2, 3]

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dim

        # Default parameter set
        self.dim = dim
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dim == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = "trilinear"
        elif dim == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = "bilinear"
        else:
            raise NotImplementedError

        # Output transform
        self.w = nn.Sequential(
            conv_nd(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1,
            ),
            bn(self.in_channels),
        )
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=self.sub_sample_kernel_size,
            stride=self.sub_sample_factor,
            bias=False,
        )
        self.phi = conv_nd(
            in_channels=self.gating_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.psi = conv_nd(
            in_channels=self.inter_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=True,
        )

        self.init_weights()

    def forward(self, x, g):
        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(
            self.phi(g),
            size=theta_x.shape[2:],
            mode=self.upsample_mode,
            align_corners=False,
        )
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(
            sigm_psi_f, size=x.shape[2:], mode=self.upsample_mode, align_corners=False
        )
        y = sigm_psi_f.expand_as(x) * x
        wy = self.w(y)

        return wy, sigm_psi_f

    def init_weights(self):
        def weight_init(m):
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif classname.find("Linear") != -1:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif classname.find("BatchNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(weight_init)


class DummyAttention(nn.Module):
    def forward(self, x, g):
        return x, None


# this is how you add this to the model...
# if attention:
#     self.attention = GridAttention(
#         in_channels=in_channels // 2, gating_channels=in_channels, dim=dim
#     )
# else:
#     self.attention = DummyAttention()
