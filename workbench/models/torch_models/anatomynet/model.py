import torch.nn as nn
from .parts import *

"""
https://github.com/wentaozhu/AnatomyNet-for-anatomical-segmentation/blob/master/src/baselineSERes18Conc.py
"""

class AnatomyNet3D(nn.Module):
    def __init__(
        self,
        n_size=6, # used to be 4 ...
        block=SEBasicBlock3D,
        upblock=UpSEBasicBlock3D,
        upblock1=UpBasicBlock3D,
        num_classes=3,
        in_channel=1,
    ):  # BasicBlock, 3
        super(AnatomyNet3D, self).__init__()
        self.inplane = 28
        self.conv1 = nn.Conv3d(
            in_channel, self.inplane, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(self.inplane)
        self.relu = nn.LeakyReLU(inplace=True)
        self.layer1 = self._make_layer(block, 30, blocks=n_size, stride=1)
        self.layer2 = self._make_layer(block, 32, blocks=n_size, stride=1)
        self.layer3 = self._make_layer(block, 34, blocks=n_size, stride=1)
        self.layer4 = upblock(34, 32, 32, stride=1)
        self.inplane = 32
        self.layer5 = self._make_layer(block, 32, blocks=n_size - 1, stride=1)
        self.layer6 = upblock(32, 30, 30, stride=1)
        self.inplane = 30
        self.layer7 = self._make_layer(block, 30, blocks=n_size - 1, stride=1)
        self.layer8 = upblock(30, 28, 28, stride=1)
        self.inplane = 28
        self.layer9 = self._make_layer(block, 28, blocks=n_size - 1, stride=1)
        self.inplane = 28
        self.layer10 = upblock1(28, 1, 14, stride=2)
        self.layer11 = nn.Sequential(  # nn.Conv3d(16, 14, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ReLU(inplace=True),
            nn.Conv3d(14, num_classes, kernel_size=3, stride=1, padding=1, bias=True)
        )
        #       self.outconv = nn.ConvTranspose3d(self.inplane, num_classes, 2, stride=2)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride))
            self.inplane = planes
        return nn.Sequential(*layers)

    def forward(self, x0):

        x = self.conv1(x0)  # 16 1/2
        x = self.bn1(x)
        x1 = self.relu(x)

        x2 = self.layer1(
            x1
        )  # 16 1/4 16 1/4 res 16 1/4 - 16 1/4 16 1/4 res 16 1/4 - 16 1/4 16 1/4 res 16 1/4
        x3 = self.layer2(
            x2
        )  # 32 1/8 32 1/8 res 32 1/8 - 32 1/8 32 1/8 res 32 1/8 - 32 1/8 32 1/8 res 32 1/8
        x4 = self.layer3(
            x3
        )  # 64 1/16 64 1/16 res 64 1/16 - 64 1/16 64 1/16 res 64 1/16 - 64 1/16 64 1/16 res 64 1/16
        #         print('x4', x4.size())
        x5 = self.layer4(
            x4, x3
        )  # 16 1/8 48 1/8 32 1/8 32 1/8 res 32 1/8 - 32 1/8 32 1/8 res 32 1/8 - 32 1/8 32 1/8 res 32 1/8
        x5 = self.layer5(x5)
        x6 = self.layer6(
            x5, x2
        )  # 8 1/4 24 1/4 16 1/4 16 1/4 res 16 1/4 - 16 1/4 16 1/4 res 16 1/4 - 16 1/4 16 1/4 res 16 1/4
        x6 = self.layer7(x6)
        x7 = self.layer8(
            x6, x1
        )  # 4 1/2 20 1/2 16 1/2 16 1/2 res 16 1/2 - 16 1/2 16 1/2 res 16 1/2 - 16 1/2 16 1/2 res 16 1/2
        x7 = self.layer9(x7)
        x8 = self.layer10(x7, x0)
        x9 = self.layer11(x8)
        #         print(x0.size(), x.size(), x1.size(), x2.size(), x3.size(), x4.size(), x5.size(), x6.size(), \
        #               x7.size(), x8.size(), x9.size())
        return x9


# model = ResNetUNET3D(SEBasicBlock3D, UpSEBasicBlock3D, UpBasicBlock3D, 2, num_classes=9+1, in_channel=1).cuda()
