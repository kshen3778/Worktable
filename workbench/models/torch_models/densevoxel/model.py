import torch
import torch.nn as nn
from .parts import *
# from torchsummary import summary
# from lib.medzoo.BaseModelClass import BaseModel

"""
Taken from https://github.com/black0017/MedicalZooPytorch/blob/master/lib/medzoo/DenseVoxelNet.py
#########
Implementation od DenseVoxelNet based on https://arxiv.org/abs/1708.00573
Hyperparameters used:
batch size = 3
weight decay = 0.0005
momentum = 0.9
lr = 0.05
#########
"""

class DenseVoxelNet(nn.Module):
    """
    Implementation based on https://arxiv.org/abs/1708.00573
    Trainable params: 1,783,408 (roughly 1.8 mentioned in the paper)
    """

    def __init__(self, in_channels=1, num_classes=3):
        super(DenseVoxelNet, self).__init__()
        num_input_features = 16
        self.dense_1_out_features = 160
        self.dense_2_out_features = 304
        self.up_out_features = 64
        self.classes = num_classes
        self.in_channels = in_channels

        self.conv_init = nn.Conv3d(in_channels, num_input_features, kernel_size=1, stride=2, padding=0, bias=False)
        self.dense_1 = DenseBlock(num_layers=12, num_input_features=num_input_features, bn_size=1, growth_rate=12)
        self.trans = Transition(self.dense_1_out_features, self.dense_1_out_features)
        self.dense_2 = DenseBlock(num_layers=12, num_input_features=self.dense_1_out_features, bn_size=1,
                                   growth_rate=12)
        self.up_block = Upsampling(self.dense_2_out_features, self.up_out_features)
        self.conv_final = nn.Conv3d(self.up_out_features, num_classes, kernel_size=1, padding=0, bias=False)
        self.transpose = nn.ConvTranspose3d(self.dense_1_out_features, self.up_out_features, kernel_size=2, padding=0,
                                            output_padding=0,
                                            stride=2)

    def forward(self, x):
        # Main network path
        x = self.conv_init(x)
        x = self.dense_1(x)
        x, t = self.trans(x)
        x = self.dense_2(x)
        x = self.up_block(x)
        y1 = self.conv_final(x)

        # Auxiliary mid-layer prediction, kind of long-skip connection
        t = self.transpose(t)
        y2 = self.conv_final(t)
        return y1, y2

    # def test(self, device='cpu'):
    #     a = torch.rand(1, self.in_channels, 8, 8, 8)
    #     ideal_out = torch.rand(1, self.classes, 8, 8, 8)
    #     summary(self.to(torch.device(device)), (self.in_channels, 8, 8, 8), device=device)
    #     b, c = self.forward(a)
    #     import torchsummaryX
    #     torchsummaryX.summary(self, a.to(device))
    #     assert ideal_out.shape == b.shape
    #     assert ideal_out.shape == c.shape
    #     print("Test DenseVoxelNet is complete")

# model = DenseVoxelNet(in_channels=1, classes=3)
# model.test()
