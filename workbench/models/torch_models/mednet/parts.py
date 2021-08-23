import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Taken from https://github.com/black0017/MedicalZooPytorch/blob/master/lib/medzoo/ResNet3DMedNet.py (MedNet3D V2)

Original paper here: https://arxiv.org/abs/1904.00625
Implementation is strongly and modified from here: https://github.com/kenshohara/3D-ResNets-PyTorch
Network architecture, taken from paper:
We adopt the ResNet family (layers with 10, 18, 34, 50, 101, 152, and 200)
we modify the backbone network as follows:
- we set the stride of the convolution kernels in blocks 3 and 4 equal to 1 to avoid down-sampling the feature maps
- we use dilated convolutional layers with rate r= 2 as suggested in [deep-lab] for the following layers for the same purpose
- we replace the fully connected layer with a 8-branch decoder, where each branch consists of a 1x1x1 convolutional kernel
 and a corresponding up-sampling layer that scale the network output up to the original dimension.
- we optimize network parameters using the cross-entropy loss with the standard SGD method,
where the learning rate is set to 0.1, momentum set to 0.9 and weight decay set to 0.001.
o = output, p = padding, k = kernel_size, s = stride, d = dilation
For transpose-conv:
o = (i -1)*s - 2*p + k + output_padding
For conv layers :
o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
"""

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    kernel_size = 3
    if dilation > 1:
        padding = find_padding(dilation, kernel_size)

    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding, dilation=dilation,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


def find_padding(dilation, kernel):
    """
    Dynamically computes padding to keep input conv size equal to the output
    for stride = 1
    :return:
    """
    return int(((kernel - 1) * (dilation - 1) + (kernel - 1)) / 2.0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TranspConvNet(nn.Module):
    """
    (segmentation)we transfer encoder part from Med3D as the feature extraction part and
    then segmented lung in whole body followed by three groups of 3D decoder layers.
    The first set of decoder layers is composed of a transposed
    convolution layer with a kernel size of(3,3,3)and a channel number of 256
    (which isused to amplify twice the feature map), and the convolutional layer with(3,3,3)kernel
    size and 128 channels.
    """

    def __init__(self, in_channels, classes):
        super().__init__()
        conv_channels = 128
        transp_channels = 256

        transp_conv_1 = nn.ConvTranspose3d(in_channels, transp_channels, kernel_size=2, stride=2)
        batch_norm_1 = nn.BatchNorm3d(transp_channels)
        relu_1 = nn.ReLU(inplace=True)
        self.transp_1 = nn.Sequential(transp_conv_1, batch_norm_1, relu_1)

        transp_conv_2 = nn.ConvTranspose3d(transp_channels, transp_channels, kernel_size=2, stride=2)
        batch_norm_2 = nn.BatchNorm3d(transp_channels)
        relu_2 = nn.ReLU(inplace=True)
        self.transp_2 = nn.Sequential(transp_conv_2, batch_norm_2, relu_2)

        transp_conv_3 = nn.ConvTranspose3d(transp_channels, transp_channels, kernel_size=2, stride=2)
        batch_norm_3 = nn.BatchNorm3d(transp_channels)
        relu_3 = nn.ReLU(inplace=True)
        self.transp_3 = nn.Sequential(transp_conv_3, batch_norm_3, relu_3)

        conv1 = conv3x3x3(transp_channels, conv_channels, stride=1, padding=1)
        batch_norm_2 = nn.BatchNorm3d(conv_channels)
        relu_2 = nn.ReLU(inplace=True)

        self.conv_1 = nn.Sequential(conv1, batch_norm_2, relu_2)
        self.conv_final = conv1x1x1(conv_channels, classes, stride=1)

    def forward(self, x):
        x = self.transp_1(x)
        x = self.transp_2(x)
        x = self.transp_3(x)
        x = self.conv_1(x)
        y = self.conv_final(x)
        return y
