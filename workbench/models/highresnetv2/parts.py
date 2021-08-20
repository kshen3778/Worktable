import torch
import torch.nn as nn

class ConvInit(nn.Module):
    def __init__(self, in_channels):
        super(ConvInit, self).__init__()
        self.num_features = 6
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=1)
        bn1 = torch.nn.BatchNorm3d(self.num_features)
        relu1 = nn.ReLU()

        self.norm = nn.Sequential(bn1, relu1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.norm(y1)

        return y1, y2


class ConvRed(nn.Module):
    def __init__(self, in_channels):
        super(ConvRed, self).__init__()
        self.num_features = 6
        self.in_channels = in_channels

        bn1 = torch.nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=1)
        self.conv_red = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_red(x)


class DilatedConv2(nn.Module):
    def __init__(self, in_channels):
        super(DilatedConv2, self).__init__()
        self.num_features = 12
        self.in_channels = in_channels
        bn1 = torch.nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=2, dilation=2)

        self.conv_dil = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_dil(x)


class DilatedConv4(nn.Module):
    def __init__(self, in_channels):
        super(DilatedConv4, self).__init__()
        self.num_features = 24
        self.in_channels = in_channels

        bn1 = torch.nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=4, dilation=4)

        self.conv_dil = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_dil(x)


class Conv1x1x1(nn.Module):
    def __init__(self, in_channels, classes):
        super(Conv1x1x1, self).__init__()
        self.num_features = classes
        self.in_channels = in_channels

        bn1 = torch.nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=1)

        self.conv_dil = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_dil(x)
