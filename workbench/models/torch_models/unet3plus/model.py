# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .parts import unetConv3, init_weights
import numpy as np


class UNet_3Plus(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=1,
        feature_scale=4,
        is_deconv=True,
        is_batchnorm=True,
        factor=1,
        leak_p=0.1,
    ):
        super(UNet_3Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [
            64 // factor,
            128 // factor,
            256 // factor,
            512 // factor,
            1024 // factor,
        ]

        ## -------------Encoder--------------
        self.conv1 = unetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = unetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = unetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = unetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        self.conv5 = unetConv3(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        """stage 4d"""
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool3d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv3d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.LeakyReLU(leak_p, inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu4d_1 = nn.LeakyReLU(leak_p, inplace=True)

        """stage 3d"""
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv3d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd4_UT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode="trilinear")  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.LeakyReLU(leak_p, inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu3d_1 = nn.LeakyReLU(leak_p, inplace=True)

        """stage 2d """
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv3d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd3_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode="trilinear")  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv3d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd4_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode="trilinear")  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.LeakyReLU(leak_p, inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu2d_1 = nn.LeakyReLU(leak_p, inplace=True)

        """stage 1d"""
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv3d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd2_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode="trilinear")  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv3d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd3_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode="trilinear")  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv3d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd4_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode="trilinear")  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.LeakyReLU(leak_p, inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu1d_1 = nn.LeakyReLU(leak_p, inplace=True)

        # output
        self.outconv1 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(
            self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1)))
        )
        h2_PT_hd4 = self.h2_PT_hd4_relu(
            self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2)))
        )
        h3_PT_hd4 = self.h3_PT_hd4_relu(
            self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3)))
        )
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(
            self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5)))
        )
        hd4 = self.relu4d_1(
            self.bn4d_1(
                self.conv4d_1(
                    torch.cat(
                        (h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1
                    )
                )
            )
        )  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(
            self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1)))
        )
        h2_PT_hd3 = self.h2_PT_hd3_relu(
            self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2)))
        )
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(
            self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4)))
        )
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(
            self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5)))
        )
        hd3 = self.relu3d_1(
            self.bn3d_1(
                self.conv3d_1(
                    torch.cat(
                        (h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1
                    )
                )
            )
        )  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(
            self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1)))
        )
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(
            self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))
        )
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(
            self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4)))
        )
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(
            self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5)))
        )
        hd2 = self.relu2d_1(
            self.bn2d_1(
                self.conv2d_1(
                    torch.cat(
                        (h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1
                    )
                )
            )
        )  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(
            self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))
        )
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(
            self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))
        )
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(
            self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4)))
        )
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(
            self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5)))
        )
        hd1 = self.relu1d_1(
            self.bn1d_1(
                self.conv1d_1(
                    torch.cat(
                        (h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1
                    )
                )
            )
        )  # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes

        return d1  # F.sigmoid(d1)


class UNet_3Plus_DeepSup(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=1,
        feature_scale=4,
        is_deconv=True,
        is_batchnorm=True,
        factor=1,
        leak_p=0.1,
    ):
        super(UNet_3Plus_DeepSup, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [
            64 // factor,
            128 // factor,
            256 // factor,
            512 // factor,
            1024 // factor,
        ]

        ## -------------Encoder--------------
        self.conv1 = unetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = unetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = unetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = unetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        self.conv5 = unetConv3(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        """stage 4d"""
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool3d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv3d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        # self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd5_UT_hd4 = nn.ConvTranspose3d(filters[4], filters[4], 2, stride=2)
        self.hd5_UT_hd4_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.LeakyReLU(leak_p, inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu4d_1 = nn.LeakyReLU(leak_p, inplace=True)

        """stage 3d"""
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        # self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd4_UT_hd3 = nn.ConvTranspose3d(self.UpChannels, self.UpChannels, 2, stride=2)
        self.hd4_UT_hd3_conv = nn.Conv3d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd4_UT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        # self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode="trilinear")  # 14*14
        self.hd5_UT_hd3 = nn.ConvTranspose3d(filters[4], filters[4], 4, stride=4)
        self.hd5_UT_hd3_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.LeakyReLU(leak_p, inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu3d_1 = nn.LeakyReLU(leak_p, inplace=True)

        """stage 2d """
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        # self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd3_UT_hd2 = nn.ConvTranspose3d(self.UpChannels, self.UpChannels, 2, stride=2)
        self.hd3_UT_hd2_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        # self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode="trilinear")  # 14*14
        self.hd4_UT_hd2 = nn.ConvTranspose3d(self.UpChannels, self.UpChannels, 4, stride=4)
        self.hd4_UT_hd2_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        # self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode="trilinear")  # 14*14
        self.hd5_UT_hd2 = nn.ConvTranspose3d(filters[4], filters[4], 8, stride=8)
        self.hd5_UT_hd2_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.LeakyReLU(leak_p, inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu2d_1 = nn.LeakyReLU(leak_p, inplace=True)

        """stage 1d"""
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        # self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd2_UT_hd1 = nn.ConvTranspose3d(self.UpChannels, self.UpChannels, 2, stride=2)
        self.hd2_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        # self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode="trilinear")  # 14*14
        self.hd3_UT_hd1 = nn.ConvTranspose3d(self.UpChannels, self.UpChannels, 4, stride=4)
        self.hd3_UT_hd1_conv = nn.Conv3d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd3_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        # self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode="trilinear")  # 14*14
        self.hd4_UT_hd1 = nn.ConvTranspose3d(self.UpChannels, self.UpChannels, 8, stride=8)
        self.hd4_UT_hd1_conv = nn.Conv3d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd4_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        # self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode="trilinear")  # 14*14
        self.hd5_UT_hd1 = nn.ConvTranspose3d(filters[4], filters[4], 16, stride=16)
        self.hd5_UT_hd1_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.LeakyReLU(leak_p, inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu1d_1 = nn.LeakyReLU(leak_p, inplace=True)

        # -------------trilinear Upsampling--------------
        # self.upscore6 = nn.Upsample(scale_factor=32, mode="trilinear")  ###
        self.upscore6 = nn.ConvTranspose3d(5, 5, 32, stride=32)
        # self.upscore5 = nn.Upsample(scale_factor=16, mode="trilinear")
        self.upscore5 = nn.ConvTranspose3d(5, 5, 16, stride=16)
        # self.upscore4 = nn.Upsample(scale_factor=8, mode="trilinear")
        self.upscore4 = nn.ConvTranspose3d(5, 5, 8, stride=8)
        # self.upscore3 = nn.Upsample(scale_factor=4, mode="trilinear")
        self.upscore3 = nn.ConvTranspose3d(5, 5, 4, stride=4)
        # self.upscore2 = nn.Upsample(scale_factor=2, mode="trilinear")
        self.upscore2 = nn.ConvTranspose3d(5, 5, 2, stride=2)

        # DeepSup
        self.outconv1 = nn.Conv3d(self.UpChannels, 5, 3, padding=1)
        self.outconv2 = nn.Conv3d(self.UpChannels, 5, 3, padding=1)
        self.outconv3 = nn.Conv3d(self.UpChannels, 5, 3, padding=1)
        self.outconv4 = nn.Conv3d(self.UpChannels, 5, 3, padding=1)
        self.outconv5 = nn.Conv3d(filters[4], 5, 3, padding=1)
        # self.outconv6 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)

        self.final_conv = nn.Conv3d(5 * 5, n_classes, kernel_size=1, bias=False)
        self.final_norm = nn.BatchNorm3d(5 * 5)
        self.final_relu = nn.PReLU(5*5)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(
            self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1)))
        )
        h2_PT_hd4 = self.h2_PT_hd4_relu(
            self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2)))
        )
        h3_PT_hd4 = self.h3_PT_hd4_relu(
            self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3)))
        )
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(
            self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5)))
        )
        hd4 = self.relu4d_1(
            self.bn4d_1(
                self.conv4d_1(
                    torch.cat(
                        (h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1
                    )
                )
            )
        )  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(
            self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1)))
        )
        h2_PT_hd3 = self.h2_PT_hd3_relu(
            self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2)))
        )
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(
            self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4)))
        )
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(
            self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5)))
        )
        hd3 = self.relu3d_1(
            self.bn3d_1(
                self.conv3d_1(
                    torch.cat(
                        (h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1
                    )
                )
            )
        )  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(
            self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1)))
        )
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(
            self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))
        )
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(
            self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4)))
        )
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(
            self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5)))
        )
        hd2 = self.relu2d_1(
            self.bn2d_1(
                self.conv2d_1(
                    torch.cat(
                        (h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1
                    )
                )
            )
        )  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(
            self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))
        )
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(
            self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))
        )
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(
            self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4)))
        )
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(
            self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5)))
        )
        hd1 = self.relu1d_1(
            self.bn1d_1(
                self.conv1d_1(
                    torch.cat(
                        (h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1
                    )
                )
            )
        )  # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)  # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)  # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256

        d1 = self.outconv1(hd1)  # 256
        d1 = torch.cat((F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5)), 1) #d1, d2, d3, d4, d5
        d1 = self.final_relu(self.final_norm(d1))
        # combine channels here so we don't have to later...
        d1 = self.final_conv(d1)

        return d1


class UNet_3Plus_DeepSup_CGM(nn.Module):
    def __init__(
        self,
        in_channels=3,
        n_classes=1,
        feature_scale=4,
        is_deconv=True,
        is_batchnorm=True,
        factor=1,
        leak_p=0.1,
    ):

        super(UNet_3Plus_DeepSup_CGM, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [
            64 // factor,
            128 // factor,
            256 // factor,
            512 // factor,
            1024 // factor,
        ]

        ## -------------Encoder--------------
        self.conv1 = unetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = unetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = unetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = unetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        self.conv5 = unetConv3(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        """stage 4d"""
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool3d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv3d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.LeakyReLU(leak_p, inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu4d_1 = nn.LeakyReLU(leak_p, inplace=True)

        """stage 3d"""
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv3d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd4_UT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode="trilinear")  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.LeakyReLU(leak_p, inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu3d_1 = nn.LeakyReLU(leak_p, inplace=True)

        """stage 2d """
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.LeakyReLU(leak_p, inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv3d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd3_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode="trilinear")  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv3d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd4_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode="trilinear")  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.LeakyReLU(leak_p, inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu2d_1 = nn.LeakyReLU(leak_p, inplace=True)

        """stage 1d"""
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode="trilinear")  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv3d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd2_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode="trilinear")  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv3d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd3_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode="trilinear")  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv3d(
            self.UpChannels, self.CatChannels, 3, padding=1
        )
        self.hd4_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.LeakyReLU(leak_p, inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode="trilinear")  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.LeakyReLU(leak_p, inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu1d_1 = nn.LeakyReLU(leak_p, inplace=True)

        # -------------trilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode="trilinear")  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode="trilinear")
        self.upscore4 = nn.Upsample(scale_factor=8, mode="trilinear")
        self.upscore3 = nn.Upsample(scale_factor=4, mode="trilinear")
        self.upscore2 = nn.Upsample(scale_factor=2, mode="trilinear")

        # DeepSup
        self.outconv1 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv6 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)

        self.final_conv = nn.Conv3d(n_classes * 5, n_classes, 1, 0)

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv3d(512, 2, 1),
            nn.AdaptiveMaxPool3d(1),
            nn.Sigmoid(),
        )

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.Conv3(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        # -------------Classification-------------
        cls_branch = self.cls(hd5).squeeze(3).squeeze(2)  # (B,N,1,1)->(B,N)
        cls_branch_max = cls_branch.argmax(dim=1)
        cls_branch_max = cls_branch_max[:, np.newaxis].float()

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(
            self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1)))
        )
        h2_PT_hd4 = self.h2_PT_hd4_relu(
            self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2)))
        )
        h3_PT_hd4 = self.h3_PT_hd4_relu(
            self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3)))
        )
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(
            self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5)))
        )
        hd4 = self.relu4d_1(
            self.bn4d_1(
                self.conv4d_1(
                    torch.cat(
                        (h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1
                    )
                )
            )
        )  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(
            self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1)))
        )
        h2_PT_hd3 = self.h2_PT_hd3_relu(
            self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2)))
        )
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(
            self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4)))
        )
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(
            self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5)))
        )
        hd3 = self.relu3d_1(
            self.bn3d_1(
                self.conv3d_1(
                    torch.cat(
                        (h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1
                    )
                )
            )
        )  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(
            self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1)))
        )
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(
            self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))
        )
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(
            self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4)))
        )
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(
            self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5)))
        )
        hd2 = self.relu2d_1(
            self.bn2d_1(
                self.conv2d_1(
                    torch.cat(
                        (h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1
                    )
                )
            )
        )  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(
            self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))
        )
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(
            self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))
        )
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(
            self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4)))
        )
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(
            self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5)))
        )
        hd1 = self.relu1d_1(
            self.bn1d_1(
                self.conv1d_1(
                    torch.cat(
                        (h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1
                    )
                )
            )
        )  # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)  # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)  # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256

        d1 = self.outconv1(hd1)  # 256

        d1 = self.dotProduct(d1, cls_branch_max)

        d_final = torch.cat((d1, d2, d3, d4, d5), 1)
        final = self.final_conv(d_final)

        return final  # F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5)
