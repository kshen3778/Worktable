"""
Taken from https://github.com/Achilleas/pytorch-mri-segmentation-3D/blob/master/architectures/deeplab_3D/deeplab_resnet_3D.py
Paper: https://arxiv.org/abs/1706.05587
"""
import torch.nn as nn
from .parts import *

affine_par = True


class MS_Deeplab(nn.Module):
    def __init__(self, block, num_classes, factor):
        super(MS_Deeplab, self).__init__()
        self.Scale = ResNet(block, [1, 1, 1, 1], num_classes, factor=factor)

    def forward(self, x):
        s0 = x.size()[2]
        s1 = x.size()[3]
        s2 = x.size()[4]
        # self.interp3 = nn.Upsample(size = ( outS(s0), outS(s1), outS(s2) ), mode= 'nearest')
        self.interp = nn.Upsample(size=(s0, s1, s2), mode="trilinear")
        out = self.interp(self.Scale(x))
        # out = []
        # add 1x1 convolution
        # add ASPP (6, 12, 18 dilations)
        # add avg max pooling with 1x1 conv
        # print('N - upsample', out.size())
        # out.append(self.Scale(x))
        return out


def Deeplabv3(num_classes=3, factor=1):
    model = MS_Deeplab(Bottleneck, num_classes, factor)
    return model
