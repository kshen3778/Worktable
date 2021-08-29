# TODO: Pre-calculate output sizes when using valid convolutions
# taken from https://elektronn3.readthedocs.io/en/latest/source/elektronn3.models.unet.html
# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Author: Martin Drawitsch

from typing import Sequence, Union
from torch import nn
from torch.utils.checkpoint import checkpoint
from .parts import *

class EK3UNet(nn.Module):
    """Modified version of U-Net, adapted for 3D biomedical image segmentation

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding, expansive pathway)
    about an input tensor is merged with information representing the
    localization of details (from the encoding, compressive pathway).

    - Original paper: https://arxiv.org/abs/1505.04597
    - Base implementation: https://github.com/jaxony/unet-pytorch


    Modifications to the original paper (@jaxony):

    - Padding is used in size-3-convolutions to prevent loss
      of border pixels.
    - Merging outputs does not require cropping due to (1).
    - Residual connections can be used by specifying
      UNet(merge_mode='add').
    - If non-parametric upsampling is used in the decoder
      pathway (specified by upmode='upsample'), then an
      additional 1x1 convolution occurs after upsampling
      to reduce channel dimensionality by a factor of 2.
      This channel halving happens with the convolution in
      the tranpose convolution (specified by upmode='transpose').

    Additional modifications (@mdraw):

    - Operates on 3D image data (5D tensors) instead of 2D data
    - Uses 3D convolution, 3D pooling etc. by default
    - Each network block pair (the two corresponding submodules in the
      encoder and decoder pathways) can be configured to either work
      in 3D or 2D mode (3D/2D convolution, pooling etc.)
      with the `planar_blocks` parameter.
      This is helpful for dealing with data anisotropy (commonly the
      depth axis has lower resolution in SBEM data sets, so it is not
      as important for convolution/pooling) and can reduce the complexity of
      models (parameter counts, speed, memory usage etc.).
      Note: If planar blocks are used, the input patch size should be
      adapted by reducing depth and increasing height and width of inputs.
    - Configurable activation function.
    - Optional normalization

    Gradient checkpointing can be used to reduce memory consumption while
    training. To make use of gradient checkpointing, just run the
    ``forward_gradcp()`` instead of the regular ``forward`` method.
    This makes the backward pass a bit slower, but the memory savings can be
    huge (usually around 20% - 50%, depending on hyperparameters). Checkpoints
    are made after each network *block*.
    See https://pytorch.org/docs/master/checkpoint.html and
    https://arxiv.org/abs/1604.06174 for more details.
    Gradient checkpointing is not supported in TorchScript mode.

    Args:
        in_channels: Number of input channels
            (e.g. 1 for single-grayscale inputs, 3 for RGB images)
            Default: 1
        out_channels: Number of output channels (in classification/semantic
            segmentation, this is the number of different classes).
            Default: 2
        n_blocks: Number of downsampling/convolution blocks (max-pooling)
            in the encoder pathway. The decoder (upsampling/upconvolution)
            pathway will consist of `n_blocks - 1` blocks.
            Increasing `n_blocks` has two major effects:

            - The network will be deeper
              (n + 1 -> 4 additional convolution layers)
            - Since each block causes one additional downsampling, more
              contextual information will be available for the network,
              enhancing the effective visual receptive field.
              (n + 1 -> receptive field is approximately doubled in each
              dimension, except in planar blocks, in which it is only
              doubled in the H and W image dimensions)

            **Important note**: Always make sure that the spatial shape of
            your input is divisible by the number of blocks, because
            else, concatenating downsampled features will fail.
        start_filts: Number of filters for the first convolution layer.
            Note: The filter counts of the later layers depend on the
            choice of `merge_mode`.
        up_mode: Upsampling method in the decoder pathway.
            Choices:

            - 'transpose' (default): Use transposed convolution
              ("Upconvolution")
            - 'resizeconv_nearest': Use resize-convolution with nearest-
              neighbor interpolation, as proposed in
              https://distill.pub/2016/deconv-checkerboard/
            - 'resizeconv_linear: Same as above, but with (bi-/tri-)linear
              interpolation
            - 'resizeconv_nearest1': Like 'resizeconv_nearest', but using a
              light-weight 1x1 convolution layer instead of a spatial convolution
            - 'resizeconv_linear1': Like 'resizeconv_nearest', but using a
              light-weight 1x1-convolution layer instead of a spatial convolution
        merge_mode: How the features from the encoder pathway should
            be combined with the decoder features.
            Choices:

            - 'concat' (default): Concatenate feature maps along the
              `C` axis, doubling the number of filters each block.
            - 'add': Directly add feature maps (like in ResNets).
              The number of filters thus stays constant in each block.

            Note: According to https://arxiv.org/abs/1701.03056, feature
            concatenation ('concat') generally leads to better model
            accuracy than 'add' in typical medical image segmentation
            tasks.
        planar_blocks: Each number i in this sequence leads to the i-th
            block being a "planar" block. This means that all image
            operations performed in the i-th block in the encoder pathway
            and its corresponding decoder counterpart disregard the depth
            (`D`) axis and only operate in 2D (`H`, `W`).
            This is helpful for dealing with data anisotropy (commonly the
            depth axis has lower resolution in SBEM data sets, so it is
            not as important for convolution/pooling) and can reduce the
            complexity of models (parameter counts, speed, memory usage
            etc.).
            Note: If planar blocks are used, the input patch size should
            be adapted by reducing depth and increasing height and
            width of inputs.
        activation: Name of the non-linear activation function that should be
            applied after each network layer.
            Choices (see https://arxiv.org/abs/1505.00853 for details):

            - 'relu' (default)
            - 'silu': Sigmoid Linear Unit (SiLU, aka Swish)
            - 'leaky': Leaky ReLU (slope 0.1)
            - 'prelu': Parametrized ReLU. Best for training accuracy, but
              tends to increase overfitting.
            - 'rrelu': Can improve generalization at the cost of training
              accuracy.
            - Or you can pass an nn.Module instance directly, e.g.
              ``activation=torch.nn.ReLU()``
        normalization: Type of normalization that should be applied at the end
            of each block. Note that it is applied after the activated conv
            layers, not before the activation. This scheme differs from the
            original batch normalization paper and the BN scheme of 3D U-Net,
            but it delivers better results this way
            (see https://redd.it/67gonq).
            Choices:

            - 'group' for group normalization (G=8)
            - 'group<G>' for group normalization with <G> groups
              (e.g. 'group16') for G=16
            - 'instance' for instance normalization
            - 'batch' for batch normalization (default)
            - 'none' or ``None`` for no normalization
        attention: If ``True``, use grid attention in the decoding pathway,
            as proposed in https://arxiv.org/abs/1804.03999.
            Default: ``False``.
        full_norm: If ``True`` (default), perform normalization after each
            (transposed) convolution in the network (which is what almost
            all published neural network architectures do).
            If ``False``, only normalize after the last convolution
            layer of each block, in order to save resources. This was also
            the default behavior before this option was introduced.
        dim: Spatial dimensionality of the network. Choices:

            - 3 (default): 3D mode. Every block fully works in 3D unless
              it is excluded by the ``planar_blocks`` setting.
              The network expects and operates on 5D input tensors
              (N, C, D, H, W).
            - 2: Every block and every operation works in 2D, expecting
              4D input tensors (N, C, H, W).
        conv_mode: Padding mode of convolutions. Choices:

            - 'same' (default): Use SAME-convolutions in every layer:
              zero-padding inputs so that all convolutions preserve spatial
              shapes and don't produce an offset at the boundaries.
            - 'valid': Use VALID-convolutions in every layer: no padding is
              used, so every convolution layer reduces spatial shape by 2 in
              each dimension. Intermediate feature maps of the encoder pathway
              are automatically cropped to compatible shapes so they can be
              merged with decoder features.
              Advantages:

              - Less resource consumption than SAME because feature maps
                have reduced sizes especially in deeper layers.
              - No "fake" data (that is, the zeros from the SAME-padding)
                is fed into the network. The output regions that are influenced
                by zero-padding naturally have worse quality, so they should
                be removed in post-processing if possible (see
                ``overlap_shape`` in :py:mod:`elektronn3.inference`).
                Using VALID convolutions prevents the unnecessary computation
                of these regions that need to be cut away anyways for
                high-quality tiled inference.
              - Avoids the issues described in https://arxiv.org/abs/1811.11718.
              - Since the network will not receive zero-padded inputs, it is
                not required to learn a robustness against artificial zeros
                being in the border regions of inputs. This should reduce the
                complexity of the learning task and allow the network to
                specialize better on understanding the actual, unaltered
                inputs (effectively requiring less parameters to fit).

              Disadvantages:

              - Using this mode poses some additional constraints on input
                sizes and requires you to center-crop your targets,
                so it's harder to use in practice than the 'same' mode.
              - In some cases it might be preferable to get low-quality
                outputs at image borders as opposed to getting no outputs at
                the borders. Most notably this is the case if you do training
                and inference not on small patches, but on complete images in
                a single step.
    """
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 2,
            n_blocks: int = 3,
            start_filts: int = 32,
            up_mode: str = 'transpose',
            merge_mode: str = 'concat',
            planar_blocks: Sequence = (),
            batch_norm: str = 'unset',
            attention: bool = False,
            activation: Union[str, nn.Module] = 'relu',
            normalization: str = 'batch',
            full_norm: bool = True,
            dim: int = 3,
            conv_mode: str = 'same',
    ):
        super().__init__()

        if n_blocks < 1:
            raise ValueError('n_blocks must be > 1.')

        if dim not in {2, 3}:
            raise ValueError('dim has to be 2 or 3')
        if dim == 2 and planar_blocks != ():
            raise ValueError(
                'If dim=2, you can\'t use planar_blocks since everything will '
                'be planar (2-dimensional) anyways.\n'
                'Either set dim=3 or set planar_blocks=().'
            )
        if up_mode in ('transpose', 'upsample', 'resizeconv_nearest', 'resizeconv_linear',
                       'resizeconv_nearest1', 'resizeconv_linear1'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for upsampling".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        # TODO: Remove merge_mode=add. It's just worse than concat
        if 'resizeconv' in self.up_mode and self.merge_mode == 'add':
            raise ValueError("up_mode \"resizeconv\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "n_blocks channels (by half).")

        if len(planar_blocks) > n_blocks:
            raise ValueError('planar_blocks can\'t be longer than n_blocks.')
        if planar_blocks and (max(planar_blocks) >= n_blocks or min(planar_blocks) < 0):
            raise ValueError(
                'planar_blocks has invalid value range. All values have to be'
                'block indices, meaning integers between 0 and (n_blocks - 1).'
            )

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.n_blocks = n_blocks
        self.normalization = normalization
        self.attention = attention
        self.conv_mode = conv_mode
        self.activation = activation
        self.dim = dim

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        if batch_norm != 'unset':
            raise RuntimeError(
                'The `batch_norm` option has been replaced with the more general `normalization` option.\n'
                'If you still want to use batch normalization, set `normalization=batch` instead.'
            )

        # Indices of blocks that should operate in 2D instead of 3D mode,
        # to save resources
        self.planar_blocks = planar_blocks

        # create the encoder pathway and add to a list
        for i in range(n_blocks):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i < n_blocks - 1 else False
            planar = i in self.planar_blocks

            down_conv = DownConv(
                ins,
                outs,
                pooling=pooling,
                planar=planar,
                activation=activation,
                normalization=normalization,
                full_norm=full_norm,
                dim=dim,
                conv_mode=conv_mode,
            )
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires n_blocks-1 blocks
        for i in range(n_blocks - 1):
            ins = outs
            outs = ins // 2
            planar = n_blocks - 2 - i in self.planar_blocks

            up_conv = UpConv(
                ins,
                outs,
                up_mode=up_mode,
                merge_mode=merge_mode,
                planar=planar,
                activation=activation,
                normalization=normalization,
                attention=attention,
                full_norm=full_norm,
                dim=dim,
                conv_mode=conv_mode,
            )
            self.up_convs.append(up_conv)

        self.conv_final = conv1(outs, self.out_channels, dim=dim)

        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, GridAttention):
            return
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if getattr(m, 'bias') is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        encoder_outs = []

        # Encoder pathway, save outputs for merging
        i = 0  # Can't enumerate because of https://github.com/pytorch/pytorch/issues/16123
        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
            i += 1

        # Decoding by UpConv and merging with saved outputs of encoder
        i = 0
        for module in self.up_convs:
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
            i += 1

        # No softmax is used, so you need to apply it in the loss.
        x = self.conv_final(x)
        # Uncomment the following line to temporarily store output for
        #  receptive field estimation using fornoxai/receptivefield:
        # self.feature_maps = [x]  # Currently disabled to save memory
        return x

    def forward_gradcp(self, x):
        """``forward()`` implementation with gradient checkpointing enabled.
        Apart from checkpointing, this behaves the same as ``forward()``."""
        encoder_outs = []
        i = 0
        for module in self.down_convs:
            x, before_pool = checkpoint(module, x)
            encoder_outs.append(before_pool)
            i += 1
        i = 0
        for module in self.up_convs:
            before_pool = encoder_outs[-(i+2)]
            x = checkpoint(module, before_pool, x)
            i += 1
        x = self.conv_final(x)
        # self.feature_maps = [x]  # Currently disabled to save memory
        return x
