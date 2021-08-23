# make a basic 3D unet architecture ... simple 3D vanilla unet ...
from torch import nn
from .additions import ProjectExciteLayer, GridAttention, DeformConv3D
import torch

# idea is to iterate through all our data. Split patients by Disease site
# save .csv image path, with center of mass of GTV...
# then for that slice take window +/- 31 slices? 63 x 256 x 256
# we will pass that through with the mask...


class ConvMe3d(nn.Module):
    def __init__(
        self,
        in_: int,
        out_: int,
        relu=True,
        deformable=False,
        kernel_size=(3, 3, 3),
        padding=(1, 1, 1),
        norm=False,
        last=False,
        drop_rate=0.15,
        layers=1,
    ):

        super().__init__()

        self.layers_ = []

        if deformable:
            self.layers_.append(DeformBlock(in_, out_, kernel_size=kernel_size[0], padding=padding[0]))
        else:
            self.layers_.append(nn.Conv3d(in_, out_, kernel_size=kernel_size, padding=padding, bias=False))
        if relu:
            self.layers_.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        else:
            self.layers_.append(nn.PReLU(out_, inplace=True))

        self.layers_.append(nn.BatchNorm3d(out_))

        if layers - 1 != 0:
            for i in range(layers - 1):

                # if deformable:
                #     self.layers_.append(DeformBlock(out_, out_, kernel_size=kernel_size[0], padding=padding[0]))
                # else:
                self.layers_.append(
                    nn.Conv3d(out_, out_, kernel_size=kernel_size, padding=padding, bias=False)
                )

                if relu:
                    self.layers_.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
                else:
                    self.layers_.append(nn.PReLU(out_, inplace=True))

                if norm == True:
                    self.layers_.append(nn.BatchNorm3d(out_))

        self.sequential = nn.Sequential(*self.layers_)
        self.drop = nn.Dropout3d(drop_rate) if drop_rate > 0 else False
        self.layers = layers
        # self.activation = nn.SELU(inplace=True)
        # use relu except for last output

    def forward(self, x):

        x = self.sequential(x)
        if self.drop is not False:
            x = self.drop(x)

        return x

class DeformBlock(nn.Module):

    def __init__(
        self,
        in_: int,
        out_: int,
        relu=True,
        kernel_size=3,
        padding=1,
    ):

        super().__init__()

        self.offsets = nn.Conv3d(in_, 81, kernel_size=kernel_size, padding=padding, bias=False)
        # define deformable conv...
        self.deform = DeformConv3D(in_, out_, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        # deformable convolution
        offsets = self.offsets(x)
        x = self.deform(x, offsets)
        return x

class EncoderBlock(nn.Module):

    """
    Link: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    --model-name 'UNET2D_2019_07_16_174443' (deep = False; no Dropout)
    """

    def __init__(
        self,
        in_: int,
        out_: int,
        kernel=(3, 3, 3),
        norm=True,
        pool=True,
        drop_rate=0.2,
        layers=1,
        project=False,
        deformable=False,
    ):
        super(EncoderBlock, self).__init__()

        self.block = ConvMe3d(
            in_, out_, kernel_size=kernel, norm=norm, layers=layers,
            drop_rate=drop_rate, deformable=deformable
        )
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2)) if pool is True else False
        self.project = ProjectExciteLayer(out_) if project is True else project
        # self.bn = nn.BatchNorm3d(out_) if norm is True else None

    def forward(self, x):

        x = self.pool(x) if self.pool is not False else x
        x = self.block(x)
        # add project and excitation layer
        x = self.project(x) if self.project is not False else x

        return x

class UpAddBlock(nn.Module):
    def __init__(
        self,
        in_: int,
        out_: int,
        selu=False,
        kernel=2,
        stride=2,
    ):
        super(UpAddBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_, out_, kernel_size=kernel, stride=stride)
        # if last is False else ConvMe3d(in_, out_, kernel_size=(1,1,1), padding=0)
        self.elu = nn.PReLU(out_, inplace=True) if selu is True else nn.LeakyReLU(0.1)

    def forward(self, x1, x2):

        out = self.elu(self.up(x1))
        out = torch.add(out, x2)

        return out

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_: int,
        out_: int,
        up_scale=True,
        conv_lay=2,
        selu=False,
        last=False,
        norm=True,
        project=False,
        attention=False,
        deformable=False,
    ):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose3d(
            in_, out_, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )  # if last is False else ConvMe3d(in_, out_, kernel_size=(1,1,1), padding=0)
        self.elu = nn.PReLU(out_, inplace=True) if selu is True else nn.LeakyReLU(0.1)
        # self.bn = nn.BatchNorm3d(out_) # added it to conv...
        self.conv = EncoderBlock(in_, out_, pool=False, norm=norm, layers=conv_lay, deformable=deformable)
        self.attention = GridAttention(in_channels=in_//2, gating_channels=in_//2) if last is False else GridAttention(in_channels=out_, gating_channels=in_//2)
        self.last_conv = ConvMe3d(out_*2, out_, kernel_size=(1, 1, 1), padding=0) if last is True else None
        self.project = ProjectExciteLayer(out_) if project is True else project
        self.up_scale = up_scale
        self.last = last
        self.norm = norm

    def forward(self, x1, x2):

        # print('Size of 1st tensor pre up conv', x1.size())
        # should be same dim size as x2
        # assumes in_ is double the channels as out...
        out = self.elu(self.up(x1))
        genc, att = self.attention(out, x2) if self.attention is not False else out
        self.att = att
        x = torch.cat([out, genc], dim=1)
        x = self.conv(x) if self.last is False else self.last_conv(x)
        # print('Size of tensor after concat:', x.size())
        x = torch.add(x, x2) if self.last is False else torch.add(x, out)
        x = self.project(x) if self.project is not False else x

        return x
