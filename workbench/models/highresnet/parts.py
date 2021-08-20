import torch.nn as nn


class HighResNetBlock(nn.Module):
    def __init__(
        self, inplanes, outplanes, padding_=1, stride=1, dilation_=1, affine_par=False
    ):
        super(HighResNetBlock, self).__init__()

        self.conv1 = nn.Conv3d(
            inplanes,
            outplanes,
            kernel_size=3,
            stride=1,
            padding=padding_,
            bias=False,
            dilation=dilation_,
        )
        self.conv2 = nn.Conv3d(
            outplanes,
            outplanes,
            kernel_size=3,
            stride=1,
            padding=padding_,
            bias=False,
            dilation=dilation_,
        )
        # 2 convolutions of same dilation. residual block
        self.bn1 = nn.BatchNorm3d(outplanes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        self.bn2 = nn.BatchNorm3d(outplanes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False

        self.relu = nn.PReLU()
        self.diff_dims = inplanes != outplanes

        self.downsample = nn.Sequential(
            nn.Conv3d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(outplanes, affine=affine_par),
        )
        for i in self.downsample._modules["1"].parameters():
            i.requires_grad = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.diff_dims:
            residual = self.downsample(residual)

        out += residual

        return out
