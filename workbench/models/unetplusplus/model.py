import torch
from torch import nn
from .parts import *

__all__ = ["VGGUNet", "NestedUNet"]


class VGGUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, leak_p=0.1, factor=1, **kwargs):
        super().__init__()

        nb_filter = [
            32 // factor,
            64 // factor,
            128 // factor,
            256 // factor,
            512 // factor,
        ]

        self.pool = nn.MaxPool3d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channels=1,
        deep_supervision=False,
        leak_p=0.1,
        factor=1,
        **kwargs,
    ):
        super().__init__()

        nb_filter = [
            40 // factor,
            80 // factor,
            160 // factor,
            320 // factor,
            640 // factor,
        ]

        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool3d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(
            nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0]
        )
        self.conv1_2 = VGGBlock(
            nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1]
        )
        self.conv2_2 = VGGBlock(
            nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2]
        )

        self.conv0_3 = VGGBlock(
            nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0]
        )
        self.conv1_3 = VGGBlock(
            nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1]
        )

        self.conv0_4 = VGGBlock(
            nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0]
        )

        if self.deep_supervision:

            self.final1 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
            self.final_bn = nn.BatchNorm3d(num_classes * 4)
            self.final_relu = nn.LeakyReLU(leak_p, inplace=True)
            self.final = nn.Conv3d(num_classes * 4, num_classes, kernel_size=1)

        else:

            self.final = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

        self.upconv1_0 = VGGBlockUP(nb_filter[1], nb_filter[1])
        self.upconv2_0 = VGGBlockUP(nb_filter[2], nb_filter[2])
        self.upconv2_1 = VGGBlockUP(nb_filter[1], nb_filter[1])
        self.upconv3_0 = VGGBlockUP(nb_filter[3], nb_filter[3])
        self.upconv3_1 = VGGBlockUP(nb_filter[2], nb_filter[2])
        self.upconv3_2 = VGGBlockUP(nb_filter[1], nb_filter[1])
        self.upconv4_0 = VGGBlockUP(nb_filter[4], nb_filter[4])
        self.upconv4_1 = VGGBlockUP(nb_filter[3], nb_filter[3])
        self.upconv4_2 = VGGBlockUP(nb_filter[2], nb_filter[2])
        self.upconv4_3 = VGGBlockUP(nb_filter[1], nb_filter[1])

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.upconv1_0(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upconv2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.upconv2_1(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upconv3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.upconv3_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.upconv3_2(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.upconv4_0(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.upconv4_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.upconv4_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.upconv4_3(x1_3)], 1))

        if self.deep_supervision:

            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)

            # added this...
            final = self.final_relu(
                self.final_bn(torch.cat((output1, output2, output3, output4), 1))
            )
            final = self.final(final)

            return final  # [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output
