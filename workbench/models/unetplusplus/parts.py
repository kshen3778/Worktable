from torch import nn


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(middle_channels)
        self.relu = nn.PReLU(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.PReLU(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out

class VGGBlockUP(nn.Module):
    def __init__(self, in_channels, out_channels, k=2):
        super().__init__()

        self.conv1 = nn.ConvTranspose3d(in_channels, out_channels, k, stride=2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.PReLU(out_channels)
        self.drop = nn.Dropout3d(0.2)
        
        # self.conv2 = nn.Conv3d(middle_channels, out_channels, 1)
        # self.bn2 = nn.BatchNorm3d(out_channels)
        # self.relu2 = nn.PReLU(out_channels)
        #

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.drop(self.relu(out))

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.drop(self.relu2(out))

        return out
