# make a basic 3D unet architecture ... simple 3D vanilla unet ...
from torch import nn
import numpy as np
from torch.nn import functional as F
import torch

class SubEncoder(nn.Module):
    def __init__(self, in_:int, out_:int, drop=False, norm=False, drop_rate=0.2):
        super().__init__()
        self.conv = ConvMe3d(in_, out_, kernel_size=(1, 1, 1), padding=0, last=True)
        self.bn = nn.BatchNorm3d(out_) if norm is True else None
        self.drop = nn.Dropout3d(drop_rate) if drop is True else None
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # crop image centrally
        x = x[:,:,x.size()[2]//2,:,:]
        x = x.unsqueeze(2)
        x = self.conv(x)
        x = self.lrelu(x)
        x = self.bn(x) if self.bn is not None else x
        x = self.drop(x) if self.drop is not None else x

        return x

class SubEncoder2(nn.Module):
    def __init__(self, in_:int, out_:int, drop_rate=0.15):
        super().__init__()
        self.conv1 = ConvMe3d(in_, out_, kernel_size=(1,5,5), padding=(0,2,2))
        self.conv2 = ConvMe3d(out_, out_, kernel_size=(1,3,3), padding=(0,1,1), last=True)
        # self.adapt = nn.AdaptiveAvgPool3d(out_)
        self.drop = nn.Dropout3d(drop_rate)

    def forward(self, x):
        # full image ...
        x = x[:,:,x.size()[2]//2,:,:]
        x = x.unsqueeze(2)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.adapt(x)
        x = self.drop(x)
        # then add to entire image...
        return x

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

    def __init__(self, in_:int, out_:int, pool=True, norm=True, selu=False, sub_enc=False, deconv=False):
        super(EncoderBlock1, self).__init__()
        self.block = nn.Sequential(
            ConvMe3d (in_, out_, kernel_size = (1, 3, 3), padding =  (0, 1, 1)),
            ConvMe3d (out_, out_, kernel_size = (1, 3, 3), padding = (0, 1, 1)),
            ConvMe3d (out_, out_, kernel_size = (1, 3, 3), padding = (0, 1, 1), last=True)
        )

        # ConvRelu3d(out_- out_//4, out_),
        # self.block2 = SubEncoder(in_)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool is True else None
        self.bn = nn.BatchNorm3d(out_) # tackle the question is batch norm best before or after RELU?
        self.elu = nn.SELU(inplace=True) if selu is True else nn.LeakyReLU(0.1, inplace=True)
        # self.selu = nn.SELU(inplace=True)
        self.sub_block = SubEncoder(in_, out_) if sub_enc else False
        self.sub_block2 = SubEncoder2(in_, out_) if sub_enc else False
        self.deconv = deconv
        self.norm = norm

    def forward(self, x):
        # print(self.pool)
        # if self.pool is not False:
        # print(self.pol)
        x = self.pool (x) if self.pool is not None else x
        x_init = self.block(x)
        x_init = self.elu(x_init)
        x_init = self.bn(x_init) if self.norm is not False else x_init

        ###############
        if self.sub_block is not False:
            x_sub = self.sub_block(x)
            # will return array (B, C, 1, x, y)
            if x_init.size()[2]//2 > 0:
                # sum to the cropped z_slice ...
                x_init[:,:,x_init.size()[2]//2,:,:] += x_sub[:,:,0,:,:]
            else:
                # assumin they're the same shape in DECONV block(s)
                # adding this to the entire image.
                x_init += x_sub

        return x_init

        # train model with second residual encoder ...
        # if self.deconv is False:
        #     # only add this block during encoding ...
        #     # before added to whole image, now add to middle Z slice...
        #     x_sub2 = self.sub_block2(self.selu(x))
        #     if x_init.size()[2]//2 > 0:
        #         # sum to the cropped z_slice ...
        #         x_init[:,:,x_init.size()[2]//2,:,:] += x_sub2
        #     else:
        #         # assumin they're the same shape in DECONV block(s)
        #         # adding this to the entire image.
        #         x_init += x_sub2

            # # assumes that nothing happens in z plane ...
            # # adding this to the entire x_init ...
            # # x_init += x_sub2

    ##############
    # x_init = self.bn(x_init) if self.norm else x_init
    # x_init = self.selu(x_init)
    #############

class EncoderBlock2(nn.Module):

    """
    Link: https://arxiv.org/pdf/1809.04430.pdf
    Have to yet make the DeepDecoderBlock class...
    """

    def __init__(self, in_:int, out_:int, norm=True, sub_enc=False, selu=True):
        super(EncoderBlock2, self).__init__()
        self.block = nn.Sequential(
            ConvMe3d(in_, out_, kernel_size =  (1, 3, 3), padding = (0, 1, 1)),
            ConvMe3d(out_, out_, kernel_size = (1, 3, 3), padding = (0, 1, 1)),
            ConvMe3d(out_, out_, kernel_size = (3, 3, 3), padding = (0, 1, 1), last=True),
        )

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))  # 1x2x2 Pooling
        self.bn = nn.BatchNorm3d(out_) if norm is True else None
        self.elu = nn.SELU(inplace=True) if selu is True else nn.LeakyReLU(0.1, inplace=True)
        self.sub_block = SubEncoder(in_, out_) if sub_enc else False

    def forward(self, x):

        x = self.pool(x) # conserve this (if sub_enc == True)
        x = self.block(x)
        x = self.elu(x) # activation
        x = self.bn(x) if self.bn is not None else x

        # if self.sub_block is not False:
        #     x_sub = self.sub_block(x)
        #     # will return array (B, C, 1, x, y)
        #     # add that to center slice of block output...
        #     x_init[:, :, x_init.size()[2]//2, :, :] += x_sub[:,:,0,:,:]

        return x

class EncoderBlock3(nn.Module):

    """
    Link: https://arxiv.org/pdf/1809.04430.pdf
    Have to yet make the DeepDecoderBlock class...
    """

    def __init__(self, in_:int, out_:int, sub_enc=False, selu=True):
        super(EncoderBlock3, self).__init__()
        # can add padded x,y convs before/after zconvs here...
        self.block = nn.Sequential(
            ConvMe3d(in_, out_, kernel_size =  (3, 1, 1)),
            ConvMe3d(out_, out_, kernel_size = (3, 1, 1)),
            ConvMe3d(out_, out_, kernel_size = (3, 1, 1), last = True)
        )

        self.pool = nn.MaxPool3d(kernel_size=(1,2,2)) # 1x2x2 Pooling
        self.bn = nn.BatchNorm3d(out_)
        self.elu = nn.SELU(inplace=True) if selu is True else nn.LeakyReLU(0.1, inplace=True)
        self.sub_block = SubEncoder(in_, out_) if sub_enc else False

    def forward(self, x):

        x = self.pool(x)
        x = self.block(x)
        x = self.elu(x)
        x = self.bn(x)

        # if self.sub_block is not False:
        #     x_sub = self.sub_block(x)
        #     # will return array (B, C, 1, x, y)
        #     # add that to center slice of block output...
        #     x_init[:, :, x_init.size()[2]//2, :, :] += x_sub[:,:,0,:,:]

        return x

# Model's main decoder class
# Default mode uses center crop in z plane after upsampling ...
# See Teranusnet for another implementation ...
class DecoderBlock1(nn.Module):
    def __init__(self, in_:int, out_:int, up_scale=True, norm=False,
                 selu=False, sub_enc=False, last=False):
        super(DecoderBlock1, self).__init__()
        self.up = nn.ConvTranspose3d(in_, in_, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # self.up_ = nn.ConvTranspose3d(in_, in_, kernel_size=(3,2,2), stride=(1, 2, 2))
        # Why would we not want to upconv in the Z plane?
        # concat on channels then in_*2
        self.elu = nn.SELU(inplace=True) if selu is True else nn.LeakyReLU(0.1, inplace=True)
        self.bn = nn.BatchNorm3d(out_) if norm is True else None
        self.conv = EncoderBlock1(in_*2, out_, pool=False, sub_enc=sub_enc, deconv=True)
        self.final = nn.Conv3d(in_*2, out_, kernel_size=(1,1,1), bias=True)
        self.up_scale = up_scale
        self.last=last

    def forward(self, x1, x2):

        if self.up_scale is True:
            x1 = self.elu(self.up(x1))
            x2_center = x2.size()[2]//2  # stack in channel(s) dimension
            x = x2 [:,:,x2_center,:,:]
            x = x.unsqueeze(2)
        else:
            x = x2

        x = torch.cat([x1, x], dim=1)
        x = self.conv(x) if self.last is True else self.final(x)
        x = self.elu(x)
        x = self.bn(x) if self.bn is not None else x

        return x
