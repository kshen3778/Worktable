import torch
import torch.nn as nn
import torch.nn.functional as F
from .parts import *


class PIPOFAN3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, factor=2):
        super(PIPOFAN3D, self).__init__()
        self.resnet = BaseResUNet(in_channels, num_classes, factor)
        # self.catconv = cat_conv(10,n_classes)
        self.att = attention(num_classes, 1)
        # this should be the size of the image...
        self.gapool1 = nn.AvgPool3d(kernel_size=(64, 192, 192))
        self.gapool2 = nn.MaxPool3d(kernel_size=(64, 192, 192))

    def forward(self, x):
        a, b, c, d, e = self.resnet(x)

        w1 = self.att(a)
        w2 = self.att(b)
        w3 = self.att(c)
        w4 = self.att(d)
        w5 = self.att(e)

        w1 = self.gapool1(w1) + self.gapool2(w1)
        w2 = self.gapool1(w2) + self.gapool2(w2)
        w3 = self.gapool1(w3) + self.gapool2(w3)
        w4 = self.gapool1(w4) + self.gapool2(w4)
        w5 = self.gapool1(w5) + self.gapool2(w5)

        w = torch.cat((w1, w2, w3, w4, w5), 1)
        w = torch.nn.Softmax()(w)

        w1 = w[:, 0:1, :, :, :]
        w2 = w[:, 1:2, :, :, :]
        w3 = w[:, 2:3, :, :, :]
        w4 = w[:, 3:4, :, :, :]
        w5 = w[:, 4:5, :, :, :]

        fi_out = w1 * a + w2 * b + w3 * c + w4 * d + w5 * e

        # softmax for uniseg
        # fi_out = F.softmax(fi_out, dim=1)

        return fi_out
