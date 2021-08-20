"""Adaptation from https://github.com/mapillary/inplace_abn ."""
from collections import OrderedDict
import torch.nn as nn
# from modules.bn import ABN
from .misc import GlobalAvgPool2d
from .residual import IdentityResidualBlock

class WiderResNet(nn.Module):
    def __init__(self,
                 structure,
                 norm_act=True,
                 classes=0,
                 factor=1):
        """Wider ResNet with pre-activation (identity mapping) blocks
        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
            of the network.
        """
        super(WiderResNet, self).__init__()
        self.structure = structure

        if len(structure) != 6:
            raise ValueError("Expected a structure with six values")

        # Initial layers
        self.mod1 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv3d(1, 16, 3, stride=1, padding=1, bias=False))
        ]))

        # Groups of residual blocks
        in_channels = 16
        channels = [(16*factor, 16*factor), (32*factor, 32*factor),
                    (64*factor, 64*factor), (64*factor, 128*factor),
                    (64*factor, 128*factor, 256*factor), (128*factor, 256*factor, 512*factor)]
        # channels = [chan*factor for chan in channels]
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                blocks.append((
                    "block%d" % (block_id + 1),
                    IdentityResidualBlock(in_channels, channels[mod_id], norm_act=norm_act)
                ))

                # Update channels and p_keep
                in_channels = channels[mod_id][-1]

            # Create module
            if mod_id <= 4:
                self.add_module("pool%d" % (mod_id + 2), nn.MaxPool3d(3, stride=2, padding=1))
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

        # Pooling and predictor
        self.bn_out = nn.Sequential(nn.BatchNorm3d(in_channels),
                                    nn.PReLU(in_channels))
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([
                ("avg_pool", GlobalAvgPool2d()),
                ("fc", nn.Linear(in_channels, classes))
            ]))

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(self.pool4(out))
        out = self.mod5(self.pool5(out))
        out = self.mod6(self.pool6(out))
        out = self.mod7(out)
        out = self.bn_out(out)

        if hasattr(self, "classifier"):
            out = self.classifier(out)

        return out
