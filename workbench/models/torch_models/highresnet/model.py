"""
Taken from https://github.com/Achilleas/pytorch-mri-segmentation-3D/blob/master/architectures/hrnet_3D/highresnet_3D.py
Paper: https://link.springer.com/chapter/10.1007/978-3-319-59050-9_28
"""

import torch.nn as nn
import torch.nn.init as init
from .parts import *

affine_par = True

class HighResNet(nn.Module):
    def __init__(self, num_classes, affine_par=affine_par):
        super(HighResNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(16, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.PReLU()

        self.block1_1 = HighResNetBlock(
            inplanes=16, outplanes=16, padding_=1, dilation_=1
        )

        self.block2_1 = HighResNetBlock(
            inplanes=16, outplanes=32, padding_=2, dilation_=2
        )
        self.block2_2 = HighResNetBlock(
            inplanes=32, outplanes=32, padding_=2, dilation_=2
        )

        self.block3_1 = HighResNetBlock(
            inplanes=32, outplanes=64, padding_=4, dilation_=4
        )
        self.block3_2 = HighResNetBlock(
            inplanes=64, outplanes=64, padding_=4, dilation_=4
        )

        self.conv2 = nn.Conv3d(64, 80, kernel_size=1, stride=1, padding=0, bias=False)
        self.upsample = nn.ConvTranspose3d(80, 80, kernel_size=2, stride=2, bias=False)
        self.conv3 = nn.Conv3d(
            80, num_classes, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x):
        paddings = (int(x.size()[2] % 2), int(x.size()[3] % 2), int(x.size()[4] % 2))
        # print('INPT SIZE', x.size())
        out = self.conv1(x)
        # print('AFTER PPOOL SIZE', out.size())
        out = self.bn1(out)
        out = self.relu(out)
        # print('A', out.size())
        # res blocks (dilation = 1)
        out = self.block1_1(out)
        # print('B', out.size())
        out = self.block1_1(out)
        # print('C', out.size())
        out = self.block1_1(out)
        # print('D', out.size())

        # res blocks (dilation = 2)
        out = self.block2_1(out)
        # print('E', out.size())
        out = self.block2_2(out)
        # print('F', out.size())
        out = self.block2_2(out)
        # print('G', out.size())

        # res blocks (dilation = 4)
        out = self.block3_1(out)
        # print('H', out.size())
        out = self.block3_2(out)
        # print('I', out.size())
        out = self.block3_2(out)
        # print('J', out.size())
        out = self.conv2(out)
        out = self.upsample(out)[:, :, paddings[0] :, paddings[1] :, paddings[2] :]
        # print('AFTER UPSAMPLE SIZE', out.size())
        out = self.conv3(out)
        # s0 = x.size()[2]
        # s1 = x.size()[3]
        # s2 = x.size()[4]
        # self.interp = nn.Upsample(size = (s0, s1, s2), mode='trilinear')
        # out = self.interp(out)
        # print('K', out.size())
        return out


def HRNet(num_classes=3):
    model = HighResNet(num_classes)
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
        elif isinstance(m, nn.Sequential):
            for m_1 in m.modules():
                if isinstance(m_1, nn.Conv3d):
                    init.kaiming_uniform(m_1.weight)
    return model
