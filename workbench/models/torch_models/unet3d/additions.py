import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable, Function
import numpy as np

"""
3D Squeeze and Excitation Modules
*****************************
3D Extensions of the following 2D squeeze and excitation blocks:
    1. `Channel Squeeze and Excitation <https://arxiv.org/abs/1709.01507>`_
    2. `Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
    3. `Channel and Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
    4. `New Project & Excite block, designed specifically for 3D inputs https://arxiv.org/pdf/1906.04649v1.pdf`
    Coded by -- Anne-Marie Rickmann (https://github.com/arickm)
    Retrieved from https://github.com/ai-med/squeeze_and_excitation/blob/master/squeeze_and_excitation/squeeze_and_excitation_3D.py
"""

class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(
            input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1)
        )

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(
            input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W)
        )

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(
            in_channels=num_channels,
            out_channels=num_channels_reduced,
            kernel_size=1,
            stride=1,
        )
        self.conv_cT = nn.Conv3d(
            in_channels=num_channels_reduced,
            out_channels=num_channels,
            kernel_size=1,
            stride=1,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum(
            [
                squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1),
            ]
        )

        # Excitation:
        final_squeeze_tensor = self.sigmoid(
            self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor)))
        )
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor


"""
Based on https://github.com/ozan-oktay/Attention-Gated-Networks
Published in https://arxiv.org/abs/1804.03999
if attention:
    self.attention = GridAttention(
        in_channels=in_channels // 2, gating_channels=in_channels, dim=dim
    )
Note: Usually called at end of upblock...
# taken from https://elektronn3.readthedocs.io/en/latest/source/elektronn3.models.unet.html
# ELEKTRONN3 - Neural Network Toolkit
"""


class GridAttention(nn.Module):
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


'''
Add 3D deformable convolution...
from https://github.com/KrakenLeaf/deform_conv_pytorch/
archive message >>
'''
# 2D
# ---------------------------------------------------------------------------------------------
# Original code of ChunhuanLin :    https://github.com/ChunhuanLin/deform_conv_pytorch
class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = Variable(torch.cat([torch.arange(0, 2*N, 2), torch.arange(1, 2*N+1, 2)]), requires_grad=False).type_as(x).long()
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # interpolation points p (Eq. 2)
        # ------------------------------
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)

        # Regular grid points q (Eq. 3)
        # -----------------------------
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        mask = mask.detach() # Detach from computational graph
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # Interpolation kernel (Eq. 4)
        # -----------------------------
        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:])) # g(qx, px) * g(qy, py)
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # Interpolation - x(q) (Eq. 3)
        # ----------------------------
        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

# 3D
# --------------------------------------------------------------------------------------------------------
class  DeformConv3D(nn.Module):
    def __init__(self, inc, outc=[], kernel_size=3, padding=1, bias=None):
        super(DeformConv3D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        #self.zero_padding = nn.functional.pad(padding)
        self.conv_kernel = nn.Conv3d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size

        # Out_channels = 3 * kernel_size_x * kernel_size_y * kernel_size_z
        M = offset.size(1)
        N = M // 3 # Number of channels

        if self.padding != 0:
            # For simplicity we pad from both sides in all 3 dimensions
            padding_use = (self.padding, self.padding, self.padding, self.padding ,self.padding, self.padding)
            x = nn.functional.pad(x, padding_use, "constant", 0)

        # Get input dimensions
        b, c, h, w, d = x.size()
        shape = (h, w, d)

        # interpolation points p (Eq. 2)
        # ------------------------------
        # (b, 3N, h, w, d)
        p = self._get_p(offset, dtype)

        # (b, h, w, d, 3N)
        p = p.contiguous().permute(0, 2, 3, 4, 1) # p = p_0 + p_n + offset

        # Use grid_sample to interpolate
        # ------------------------------
        for ii in range(N):
            # Normalize flow field to rake values in the range [-1, 1]
            flow = p[..., [t for t in range(ii, M, N)]]
            for jj in range(3):
                flow[..., jj] = 2 * flow[..., jj] / (shape[jj] - 1) - 1

            # Push through the spatial transformer
            tmp = nn.functional.grid_sample(input=x, grid=flow, mode='bilinear', padding_mode='border').contiguous()
            tmp = tmp.unsqueeze(dim=-1)

            # Aggregate
            if ii == 0:
                xt = tmp
            else:
                xt = torch.cat((xt, tmp), dim=-1)

        # For simplicity, ks is a scalar, implying kernel has same dimensions in all directions
        x_offset = self._reshape_x_offset(xt, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y, p_n_z = np.meshgrid(range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                          range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                          range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                          indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten(), p_n_z.flatten()))
        p_n = np.reshape(p_n, (1, 3*N, 1, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, d, N, dtype):
        #p_0_x, p_0_y, p_0_z = np.meshgrid(range(1, h + 1), range(1, w + 1), range(1, d + 1), indexing='ij') # 1,...,N
        p_0_x, p_0_y, p_0_z = np.meshgrid(range(0, h), range(0, w), range(0, d), indexing='ij') # 0,...N-1
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_z = p_0_z.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y, p_0_z), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w, d = offset.size(1)//3, offset.size(2), offset.size(3), offset.size(4)

        # (1, 3N, 1, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 3N, h, w, d)
        p_0 = self._get_p_0(h, w, d, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, d, _ = q.size()
        ny = x.size(3) # Padded dimension y
        nz = x.size(4)  # Padded dimension z
        c = x.size(1) # Number of channels in input x
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, d, N)
        offset_x = q[..., :N]
        offset_y = q[..., N:2*N]
        offset_z = q[..., 2*N:]
        # Convert subscripts to linear indices (i.e. Matlab's sub2ind)
        index = offset_x * ny * nz + offset_y * nz + offset_z
        # (b, c, h*w*d*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, d, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        '''
        This function arranges the interpolated x values in consecutive 3d blocks of size
        kernel_size x kernel_size x kernel_size. Since the Conv3d stride is equal to kernel_size, the convolution
        will happen only for the offset cubes and output the results in the proper locations
        Note: We assume kernel size is the same for all dimensions (cube)
        '''
        b, c, h, w, d, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks*ks].contiguous().view(b, c, h, w, d*ks*ks) for s in range(0, N, ks*ks)], dim=-1)
        N = x_offset.size(4)
        x_offset = torch.cat([x_offset[..., s:s + d*ks*ks].contiguous().view(b, c, h, w*ks, d*ks) for s in range(0, N, d*ks*ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks, d*ks)

        return x_offset


# Alternative realization
# ----------------------------
# This realization directly extends the 2D vewrsion. However, for large dimensions this approach is very memory consuming, although
# faster than the previous approach
class DeformConv3D_alternative(nn.Module):
    def __init__(self, inc, outc=[], kernel_size=3, padding=1, bias=None):
        super(DeformConv3D_alternative, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        #self.zero_padding = nn.functional.pad(padding)
        self.conv_kernel = nn.Conv3d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size

        # Out_channels = 3 * kernel_size_x * kernel_size_y * kernel_size_z
        N = offset.size(1) // 3 # Number of channels

        if self.padding != 0:
            # For simplicity we pad from both sides in all 3 dimensions
            padding_use = (self.padding, self.padding, self.padding, self.padding ,self.padding, self.padding)
            x = nn.functional.pad(x, padding_use, "constant", 0)

        # interpolation points p (Eq. 2)
        # ------------------------------
        # (b, 3N, h, w, d)
        p = self._get_p(offset, dtype)

        # (b, h, w, d, 3N)
        p = p.contiguous().permute(0, 2, 3, 4, 1) # p = p_0 + p_n + offset

        # Regular grid points q (Eq. 3)
        # -----------------------------
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        # Enumerate all integral locations in the feature map x
        # Clamp values between 0 and size of input, per each direction XYZ
        # OS: lt - Left/Top, rt - Right/Top, lb - Left/Bottom, rb - Right/Bottom?
        q_000 = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), # x.size() = b X c X h X w X d
                          torch.clamp(q_lt[..., N:2*N], 0, x.size(3) - 1),
                          torch.clamp(q_lt[..., 2*N:], 0, x.size(4) - 1)
                          ], dim=-1).long()
        q_111 = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                          torch.clamp(q_rb[..., N:2*N], 0, x.size(3) - 1),
                          torch.clamp(q_rb[..., 2*N:], 0, x.size(4) - 1)
                          ], dim=-1).long()
        q_001 = torch.cat([q_000[..., :N], q_000[..., N:2 * N], q_111[..., 2 * N:]], dim=-1)
        q_010 = torch.cat([q_000[..., :N], q_111[..., N:2 * N], q_000[..., 2 * N:]], dim=-1)
        q_011 = torch.cat([q_000[..., :N], q_111[..., N:2 * N], q_111[..., 2 * N:]], dim=-1)
        q_100 = torch.cat([q_111[..., :N], q_000[..., N:2 * N], q_000[..., 2 * N:]], dim=-1)
        q_101 = torch.cat([q_111[..., :N], q_000[..., N:2 * N], q_111[..., 2 * N:]], dim=-1)
        q_110 = torch.cat([q_111[..., :N], q_111[..., N:2 * N], q_000[..., 2 * N:]], dim=-1)

        # (b, h, w, d, N)
        mask = torch.cat([p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
                          p[..., N:2*N].lt(self.padding) + p[..., N:2*N].gt(x.size(3) - 1 - self.padding),
                          p[..., 2*N:].lt(self.padding) + p[..., 2*N:].gt(x.size(4) - 1 - self.padding)
                          ], dim=-1).type_as(p)
        mask = mask.detach() # Detach from computational graph
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1),
                       torch.clamp(p[..., N:2*N], 0, x.size(3) - 1),
                       torch.clamp(p[..., 2*N:], 0, x.size(4) - 1)
                       ], dim=-1)

        # Interpolation kernel - x(q) (Eq. 4)
        # -----------------------------------
        # bilinear kernel (b, h, w, d, N)
        g_000 = (1 + (q_000[..., :N].type_as(p) - p[..., :N])) * (1 + (q_000[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_000[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_111 = (1 - (q_111[..., :N].type_as(p) - p[..., :N])) * (1 - (q_111[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_111[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_001 = (1 + (q_000[..., :N].type_as(p) - p[..., :N])) * (1 + (q_000[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_111[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_010 = (1 + (q_000[..., :N].type_as(p) - p[..., :N])) * (1 - (q_111[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_000[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_011 = (1 + (q_000[..., :N].type_as(p) - p[..., :N])) * (1 - (q_111[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_111[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_100 = (1 - (q_111[..., :N].type_as(p) - p[..., :N])) * (1 + (q_000[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_000[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_101 = (1 - (q_111[..., :N].type_as(p) - p[..., :N])) * (1 + (q_000[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 - (q_111[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_110 = (1 - (q_111[..., :N].type_as(p) - p[..., :N])) * (1 - (q_111[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (1 + (q_000[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

        # Interpolation - x(q) (Eq. 3)
        # ----------------------------
        # (b, c, h, w, d, N)
        x_q_000 = self._get_x_q(x, q_000, N)
        x_q_111 = self._get_x_q(x, q_111, N)
        x_q_001 = self._get_x_q(x, q_001, N)
        x_q_010 = self._get_x_q(x, q_010, N)
        x_q_011 = self._get_x_q(x, q_011, N)
        x_q_100 = self._get_x_q(x, q_100, N)
        x_q_101 = self._get_x_q(x, q_101, N)
        x_q_110 = self._get_x_q(x, q_110, N)

        # (b, c, h, w, d, N)
        x_offset = g_000.unsqueeze(dim=1) * x_q_000 + \
                   g_111.unsqueeze(dim=1) * x_q_111 + \
                   g_001.unsqueeze(dim=1) * x_q_001 + \
                   g_010.unsqueeze(dim=1) * x_q_010 + \
                   g_011.unsqueeze(dim=1) * x_q_011 + \
                   g_100.unsqueeze(dim=1) * x_q_100 + \
                   g_101.unsqueeze(dim=1) * x_q_101 + \
                   g_110.unsqueeze(dim=1) * x_q_110

        # For simplicity, ks is a scalar, implying kernel has same dimensions in all directions
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y, p_n_z = np.meshgrid(range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                          range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                          range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                          indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten(), p_n_z.flatten()))
        p_n = np.reshape(p_n, (1, 3*N, 1, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, d, N, dtype):
        p_0_x, p_0_y, p_0_z = np.meshgrid(range(1, h+1), range(1, w+1), range(1, d+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_z = p_0_z.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y, p_0_z), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w, d = offset.size(1)//3, offset.size(2), offset.size(3), offset.size(4)

        # (1, 3N, 1, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 3N, h, w, d)
        p_0 = self._get_p_0(h, w, d, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, d, _ = q.size()
        ny = x.size(3) # Padded dimension y
        nz = x.size(4)  # Padded dimension z
        c = x.size(1) # Number of channels in input x
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, d, N)
        offset_x = q[..., :N]
        offset_y = q[..., N:2*N]
        offset_z = q[..., 2*N:]
        # Convert subscripts to linear indices (i.e. Matlab's sub2ind)
        index = offset_x * ny * nz + offset_y * nz + offset_z
        # (b, c, h*w*d*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, d, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        '''
        This function arranges the interpolated x values in consecutive 3d blocks of size
        kernel_size x kernel_size x kernel_size. Since the Conv3d stride is equal to kernel_size, the convolution
        will happen only for the offset cubes and output the results in the proper locations
        Note: We assume kernel size is the same for all dimensions (cube)
        '''
        b, c, h, w, d, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks*ks].contiguous().view(b, c, h, w, d*ks*ks) for s in range(0, N, ks*ks)], dim=-1)
        N = x_offset.size(4)
        x_offset = torch.cat([x_offset[..., s:s + d*ks*ks].contiguous().view(b, c, h, w*ks, d*ks) for s in range(0, N, d*ks*ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks, d*ks)

        return x_offset

# # TESTS #
# # --------------------------------------------------------------------------------------
# if __name__ == '__main__':
#     # 2D test
#     # -------------------------
#     input_2d = torch.rand(1, 1, 6, 6).cuda() # Batch X Channels X Height X Width
#     offsets_2d = nn.Conv2d(1, 18, kernel_size=3, padding=1).cuda()
#     conv_2d = DeformConv2D(1, 4, kernel_size=3, padding=1).cuda()
#
#     offs_2d = offsets_2d(input_2d)
#     output_2d = conv_2d(input_2d, offs_2d)
#     output_2d.backward(output_2d.data)
#     print(output_2d.size())
#
#     # 3D test
#     # -------------------------
#     input_3d = torch.rand(10, 4, 6, 7, 5).cuda()  # Batch X Channels X Height X Width X Depth
#     offsets_3d = nn.Conv3d(4, 81, kernel_size=3, padding=1).cuda() # Out_channels = 3 * kernel_size_x * kernel_size_y * kernel_size_z
#     conv_3d = DeformConv3D(4, 4, kernel_size=3, padding=1).cuda()
#
#     offs_3d = offsets_3d(input_3d)
#     output_3d = conv_3d(input_3d, offs_3d)
#     output_3d.backward(output_3d.data)
#     print(output_3d.size())
