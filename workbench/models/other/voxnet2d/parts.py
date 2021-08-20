2# make a basic 3D unet architecture ... simple 3D vanilla unet ...
from torch import nn
from torch.nn import functional as F
import torch

class ConvMe3d(nn.Module):
    def __init__(self, in_: int, out_:int, relu=True, kernel_size=(3, 3, 3), padding = 0, last=False, drop=False, drop_rate=0.15):
        super().__init__()
        self.conv3d = nn.Conv3d(in_, out_, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.ReLU(inplace=True) if relu else nn.SELU(inplace=True)
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

class VoxConv(nn.Module):

    def __init__(self, in_: int, out_:int, selu=True, kernel_size=(1, 3, 3), padding = (0,1,1)):
        super().__init__()
        self.bn = nn.BatchNorm3d(out_)
        self.activation = nn.SELU(inplace=True) if selu else nn.LeakyReLU(inplace=True)
        self.conv3d = nn.Sequential(
                        ConvMe3d(in_, out_, kernel_size=kernel_size, padding=padding),
                        ConvMe3d(out_, out_, kernel_size=(3,1,1), padding=(1,0,0), last=True)
                        )

        # self.activation = nn.SELU(inplace=True)
        # use relu except for last output

    def forward(self, x):
        # have to add a dimention for 3D convolution
        x = self.conv3d(x)
        x = self.activation(x)
        x = self.bn(x)

        return x


class DenseBlock(nn.Module):
    def __init__(self, in_: int, out_: int, mid_ker=(3,1,1), mid_pad=0, dr=0.1):
        super(DenseBlock, self).__init__()

        self.conv1 = VoxConv(in_, out_)
        self.conv2 = VoxConv(in_ + out_, out_)
        self.conv3 = VoxConv(in_ + out_*2, out_)
        self.conv4 = VoxConv(in_ + out_*3, out_)
        self.conv5 = VoxConv(in_ + out_*4, out_)
        self.conv6 = VoxConv(in_ + out_*5, out_)
        self.drop = nn.Dropout3d(dr)
        self.mid_block = nn.Sequential(
                        ConvMe3d(in_ + out_*6, in_ + out_*6, kernel_size=mid_ker, padding=mid_pad),
                        nn.BatchNorm3d(in_ + out_*6),
                        )
        # Output should be 28 channels in axis 1 (Conv2)
        # Output should be 172 channelse in axis 1 (Conv15)

    def forward(self, x):

        x1 = self.drop(self.conv1(x))
        x = torch.cat([x1, x], dim=1)
        x1 = self.drop(self.conv2(x))
        x = torch.cat([x1, x], dim=1)
        x1 = self.drop(self.conv3(x))
        x = torch.cat([x1, x], dim=1)
        x1 = self.drop(self.conv4(x))
        x = torch.cat([x1, x], dim=1)
        x1 = self.drop(self.conv5(x))
        x = torch.cat([x1, x], dim=1)
        x1 = self.drop(self.conv6(x))
        x = torch.cat([x1, x], dim=1)
        x = self.drop(self.mid_block(x))

        # Should we just implement regular 3D model??
        # if pool=False will return the same x/y input dim...

        return x
