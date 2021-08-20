"""The network definition that was used for a second place solution at the DeepGlobe Building Detection challenge."""
import torch
from torch import nn
from torch.nn import Sequential
from collections import OrderedDict
# from .modules.bn import ABN
from .modules.wider_resnet import WiderResNet
from .parts import *

class TernausNetV2(nn.Module):
    """Variation of the UNet architecture with InplaceABN encoder."""

    def __init__(self, num_classes=3, num_filters=16, is_deconv=True,
                 num_input_channels=1, factor=1, **kwargs):
        """
        Args:
            num_classes: Number of output classes.
            num_filters:
            is_deconv:
                True: Deconvolution layer is used in the Decoder block.
                False: Upsampling layer is used in the Decoder block.
            num_input_channels: Number of channels in the input images.
        """
        super(TernausNetV2, self).__init__()

        if 'norm_act' not in kwargs:
            norm_act = True
            # used to be activated batch normalization
            # implementation failed here...
        else:
            norm_act = kwargs['norm_act']

        self.pool = nn.MaxPool3d(2, 2)

        encoder = WiderResNet(structure=[3, 3, 6, 3, 1, 1], classes=18,
                              norm_act=norm_act, factor=factor)

        self.conv1 = Sequential(
            OrderedDict([('conv1', nn.Conv3d(num_input_channels, 16, 3, padding=1, bias=False))]))

        self.conv2 = encoder.mod2
        self.conv3 = encoder.mod3
        self.conv4 = encoder.mod4
        self.conv5 = encoder.mod5

        self.center = DecoderBlock(128*factor, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec5 = DecoderBlock(128*factor + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec4 = DecoderBlock(256 + 64*factor, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec3 = DecoderBlock(32*factor + num_filters * 8, num_filters * 4, num_filters * 4, is_deconv=is_deconv)
        self.dec2 = DecoderBlock(16*factor + num_filters * 4, num_filters * 2, num_filters, is_deconv=is_deconv)
        self.dec1 = ConvRelu(16 + num_filters, num_filters)
        self.final = nn.Conv3d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        # print('conv1: ', conv1.size())
        conv2 = self.conv2(self.pool(conv1))
        # print('conv2: ', conv2.size())
        conv3 = self.conv3(self.pool(conv2))
        # print('conv3: ', conv3.size())
        conv4 = self.conv4(self.pool(conv3))
        # print('conv4: ', conv4.size())
        conv5 = self.conv5(self.pool(conv4))
        # print('conv5: ', conv5.size())

        center = self.center(self.pool(conv5))
        # print('center: ', center.size())

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        # print('dec5: ', dec5.size())

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        final = self.final(dec1)
        # print('Final: ', final.size())
        return final
