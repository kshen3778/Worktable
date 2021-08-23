# make a basic 3D unet architecture ... simple 3D vanilla unet ...
from torch import nn
import numpy as np
from torch.nn import functional as F
import torch

class ConvMe3d(nn.Module):
    def __init__(self, in_: int, out_:int, relu=True, kernel_size=(3, 3, 3), padding = 0, last=False, drop=True, drop_rate=0.1):
        super().__init__()
        self.conv3d = nn.Conv3d(in_, out_, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.LeakyReLU(0.1) if relu else nn.SELU(inplace=True)
        self.is_last = last
        self.to_drop = drop
        self.drop = nn.Dropout3d(drop_rate)
        # self.activation = nn.SELU(inplace=True)
        # use relu except for last output

    def forward(self, x):
        x = self.conv3d(x)
        # add RELU baby ...
        x = self.activation(x) if self.is_last is False else x
        # add dropout
        x = self.drop(x) if self.to_drop is True else x
        return x

class EncoderBlock1(nn.Module):

    """
    Link: https://arxiv.org/pdf/1809.04430.pdf
    Have to yet make the DeepDecoderBlock class...
    # original block 3x3x3 convolutions...
    """

    def __init__(self, in_:int, out_:int, pool=True, norm=True, selu=False, deconv=False):
        super(EncoderBlock1, self).__init__()
        self.block = nn.Sequential(
            ConvMe3d (in_,  out_, kernel_size = (1, 3, 3), padding = (0, 1, 1)),
            ConvMe3d (out_, out_, kernel_size = (1, 3, 3), padding = (0, 1, 1)),
            ConvMe3d (out_, out_, kernel_size = (1, 3, 3), padding = (0, 1, 1)),
            ConvMe3d (out_, out_, kernel_size = (1, 3, 3), padding = (0, 1, 1), last=True)
        )

        # maxpool if true
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool is True else None
        self.bn = nn.BatchNorm3d(out_) # tackle the question is batch norm best before or after RELU?
        self.elu = nn.SELU(inplace=True) if selu is True else nn.LeakyReLU(0.1)
        # turn batch normalization on...
        self.norm = norm

    def forward(self, x):

        x = self.pool (x) if self.pool is not None else x
        x_init = self.block(x)
        x_init = self.elu(x_init)
        x_init = self.bn(x_init) if self.norm is not False else x_init

        return x_init

class EncoderBlock2(nn.Module):

    """
    Link: https://arxiv.org/pdf/1809.04430.pdf
    Have to yet make the DeepDecoderBlock class...
    """

    def __init__(self, in_:int, out_:int, selu=False, norm=True):
        super(EncoderBlock2, self).__init__()
        self.block = nn.Sequential(
            ConvMe3d(in_, out_, kernel_size =  (1, 3, 3), padding = (0, 1, 1)),
            ConvMe3d(out_, out_, kernel_size = (1, 3, 3), padding = (0, 1, 1)),
            ConvMe3d(out_, out_, kernel_size = (1, 3, 3), padding = (0, 1, 1)),
            ConvMe3d(out_, out_, kernel_size = (1, 3, 3), padding = (0, 1, 1)),
            ConvMe3d(out_, out_, kernel_size = (3, 3, 3), padding = (0, 1, 1), last=True),
        )

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))  # 1x2x2 Pooling
        self.bn = nn.BatchNorm3d(out_)
        self.elu = nn.SELU(inplace=True) if selu is True else nn.LeakyReLU(0.1)
        self.norm = norm

    def forward(self, x):

        x = self.pool(x)
        x_init = self.block(x)
        x_init = self.elu(x_init)
        x_init = self.bn(x_init) if self.norm is True else x_init

        return x_init

class EncoderBlock3(nn.Module):

    """
    Link: https://arxiv.org/pdf/1809.04430.pdf
    Have to yet make the DeepDecoderBlock class...
    """

    def __init__(self, in_:int, out_:int, selu=False, norm=True):
        super(EncoderBlock3, self).__init__()
        # can add padded x,y convs before/after zconvs here...
        self.block = nn.Sequential(
            ConvMe3d(in_, out_, kernel_size =  (3, 1, 1)),
            ConvMe3d(out_, out_, kernel_size = (3, 1, 1)),
            ConvMe3d(out_, out_, kernel_size = (3, 1, 1), last = True)
        )

        self.pool = nn.MaxPool3d(kernel_size=(1,2,2)) # 1x2x2 Pooling
        self.bn = nn.BatchNorm3d(out_)
        self.elu = nn.SELU(inplace=True) if selu is True else nn.LeakyReLU(0.1)
        self.norm=norm

    def forward(self, x):

        x = self.pool(x)
        x_init = self.block(x)
        x_init = self.elu(x_init)
        x_init = self.bn(x_init) if self.norm is True else x_init

        return x_init

# Model's main decoder class
# Default mode uses center crop in z plane after upsampling ...
# See Teranusnet for another implementation ...
class DecoderBlock1(nn.Module):
    def __init__(self, in_:int, out_:int, up_scale=True, selu=False, mode='default', last=False, norm=True):
        super(DecoderBlock1, self).__init__()
        self.up = nn.ConvTranspose3d(in_, in_, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # self.up_ = nn.ConvTranspose3d(in_, in_, kernel_size=(3,2,2), stride=(1, 2, 2))
        # Why would we not want to upconv in the Z plane?
        # concat on channels then in_*2
        self.elu = nn.SELU(inplace=True) if selu is True else nn.LeakyReLU(0.1)
        self.bn = nn.BatchNorm3d(out_)
        self.conv = EncoderBlock1(in_*2, out_, pool=False, deconv=True)
        self.last_conv = ConvMe3d(in_*2, out_, kernel_size=(1,1,1))
        self.mode = mode
        self.up_scale = up_scale
        self.last=last
        self.norm = norm

    def forward(self, x1, x2):

        # upsampling
        x1 = self.elu(self.up(x1))
        # cropping volume from skip connection
        x2_center = x2.size()[2]//2  # stack in channel(s) dimension
        x = x2[:, :, x2_center, :, :]
        x = x.unsqueeze(2)
        # completes skip connection (concatonation)
        x = torch.cat([x1, x], dim=1)
        # apply EncoderBlock1 convolutional filter
        x = self.conv(x) if self.last is False else self.last_conv(x)
        x = self.elu(x)
        x = self.bn(x) if self.norm is True else x

        return x
