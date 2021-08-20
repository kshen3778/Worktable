import torch
import torch.nn as nn
from .parts import *
from functools import partial
import torch.nn.functional as F
# from torchsummary import summary

class ResNetMed3D(nn.Module):

    def __init__(self, in_channels=3, classes=10,
                 block=BasicBlock,
                 layers=[1, 1, 1, 1],
                 block_inplanes=[64, 128, 256, 512],
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0):

        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(in_channels,
                               self.in_planes,
                               kernel_size=(7, 7, 7),
                               stride=(2, 2, 2),
                               padding=(3, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)

        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=1, dilation=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=1, dilation=4)

        self.segm = TranspConvNet(in_channels=512 * block.expansion, classes=classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride, dilation=dilation,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        print(x.shape)

        x = self.segm(x)

        return x

    # def test(self):
    #     a = torch.rand(1, self.in_channels, 16, 16, 16)
    #     y = self.forward(a)
    #     target = torch.rand(1, self.classes, 16, 16, 16)
    #     assert a.shape == y.shape

def generate_resnet3d(in_channels=1, classes=2, model_depth=18, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    res_net_dict = {10: [1, 1, 1, 1], 18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3],
                    152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}

    in_planes = [64, 128, 256, 512]

    if model_depth == 10:
        model = ResNetMed3D(in_channels=in_channels, classes=classes, block=BasicBlock,
                            layers=res_net_dict[model_depth], block_inplanes=in_planes, **kwargs)
    elif model_depth == 18:
        model = ResNetMed3D(in_channels=in_channels, classes=classes, block=BasicBlock,
                            layers=res_net_dict[model_depth], block_inplanes=in_planes, **kwargs)
    elif model_depth == 34:
        model = ResNetMed3D(in_channels=in_channels, classes=classes, block=BasicBlock,
                            layers=res_net_dict[model_depth], block_inplanes=in_planes, **kwargs)
    elif model_depth == 50:
        model = ResNetMed3D(in_channels=in_channels, classes=classes, block=Bottleneck,
                            layers=res_net_dict[model_depth], block_inplanes=in_planes, **kwargs)
    elif model_depth == 101:
        model = ResNetMed3D(in_channels=in_channels, classes=classes, block=Bottleneck,
                            layers=res_net_dict[model_depth], block_inplanes=in_planes, **kwargs)
    elif model_depth == 152:
        model = ResNetMed3D(in_channels=in_channels, classes=classes, block=Bottleneck,
                            layers=res_net_dict[model_depth], block_inplanes=in_planes, **kwargs)
    elif model_depth == 200:
        model = ResNetMed3D(in_channels=in_channels, classes=classes, block=Bottleneck,
                            layers=res_net_dict[model_depth], block_inplanes=in_planes, **kwargs)

    return model
