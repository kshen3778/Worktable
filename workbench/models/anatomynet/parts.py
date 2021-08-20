# sub-parts of the U-Net model
import torch as t
from torch import nn
import torch.nn.functional as F
from scipy.spatial.distance import dice

def conv3x3x3(in_planes, out_planes, stride=1):
    "3x3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock3D(nn.Module):
    def __init__(self, inplanes, planes, stride=1):

        super(BasicBlock3D, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm3d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
#         print(x.size())
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
#         print(x.size(), residual.size(), out.size())
        out += residual
        out = self.relu(out)
        return out

def Deconv3x3x3(in_planes, out_planes, stride=2):
    "3x3x3 deconvolution with padding"
    return nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=stride)

class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=15):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.LeakyReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class SEBasicBlock3D(nn.Module):
    # expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=15):
        super(SEBasicBlock3D, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.se = SELayer3D(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm3d(planes))
        else:
            # edited from original AnatomyNet because of
            # AttributeError: Can't pickle local object 'SEBasicBlock3D.__init__.<locals>.<lambda>'
            self.downsample = None

        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
#         if self.downsample is not None:
        residual = self.downsample(x) if self.downsample is not None else x
        out += residual
        out = self.relu(out)
        return out

class UpSEBasicBlock3D(nn.Module):
    def __init__(self, inplanes1, inplanes2, planes, stride=1, downsample=None, reduction=16):
        super(UpSEBasicBlock3D, self).__init__()
        inplanes3 = inplanes1 + inplanes2
        if stride == 2:
            self.deconv1 = Deconv3x3x3(inplanes1, inplanes1//2)
            inplanes3 = inplanes1 // 2 + inplanes2
        self.stride = stride
        # self.conv1x1x1 = nn.Conv3d(inplanes2, planes, kernel_size=1, stride=1)#, padding=1)
        self.conv1 = conv3x3x3(inplanes3, planes)#, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.se = SELayer3D(planes, reduction)
        if inplanes3 != planes:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes3, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm3d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride
    def forward(self, x1, x2):
#         print(x1.size(), x2.size())
        if self.stride == 2: x1 = self.deconv1(x1)
        # x2 = self.conv1x1x1(x2)
        #print(x1.size(), x2.size())
        out = t.cat([x1, x2], dim=1) #x1 + x2
        residual = self.downsample(out)
        #print(residual.size(), x1.size(), x2.size())
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        #print(out.size(), residual.size())
        out += residual
        out = self.relu(out)
        return out

class UpBasicBlock3D(nn.Module):

    def __init__(self, inplanes1, inplanes2, planes, stride=2):
        super(UpBasicBlock3D, self).__init__()
        inplanes3 = inplanes1 + inplanes2
        if stride == 2:
            self.deconv1 = Deconv3x3x3(inplanes1, inplanes1//2)
            inplanes3 = inplanes1//2 + inplanes2
        self.stride = stride
        # elif inplanes1 != planes:
            # self.deconv1 = nn.Conv3d(inplanes1, planes, kernel_size=1, stride=1)
        # self.conv1x1x1 = nn.Conv3d(inplanes2, planes, kernel_size=1, stride=1)#, padding=1)
        self.conv1 = conv3x3x3(inplanes3, planes)#, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        if inplanes3 != planes:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes3, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm3d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x1, x2):
#         print(x1.size(), x2.size())
        if self.stride == 2: x1 = self.deconv1(x1)
        #print(self.stride, x1.size(), x2.size())
        out = t.cat([x1, x2], dim=1)
        residual = self.downsample(out)
        #print(out.size(), residual.size())
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out
