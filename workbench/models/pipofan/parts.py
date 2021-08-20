import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class one_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(one_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(in_ch, out_ch, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class res_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(res_conv, self).__init__()
        self.conv1 = double_conv(in_ch, out_ch)
        self.bridge = one_conv(in_ch, out_ch)

    def forward(self, x):
        x1 = self.conv1(x)
        if x.shape == x1.shape:
            r = x + x1
        else:
            r = self.bridge(x) + x1
        return r


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.1),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.pool = nn.MaxPool3d(2)
        self.mpconv = double_conv(in_ch, out_ch)
        self.bridge = one_conv(in_ch, out_ch)

    def forward(self, x, y):
        x = self.pool(x)
        x_1 = torch.cat((x, y), 1)
        x_2 = self.mpconv(x_1)
        if x_1.shape == x_2.shape:
            x = x_1 + x_2
        else:
            x = self.bridge(x_1) + x_2
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)
        self.bridge = one_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[3] - x2.size()[3]
        diffY = x1.size()[4] - x2.size()[4]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x) + self.bridge(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_ch), nn.LeakyReLU(0.1), nn.Conv3d(in_ch, out_ch, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class attention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(attention, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_ch, out_ch, 1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class BaseResUNet(nn.Module):
    def __init__(self, in_channels, n_classes, factor):
        super(BaseResUNet, self).__init__()
        self.inc = inconv(in_channels, 54 // factor)
        self.dbconv1 = res_conv(54 // factor, 108 // factor)
        self.down1 = down(108 // factor, 108 // factor)
        self.dbconv2 = res_conv(54 // factor, 108 // factor)
        self.dbconv3 = res_conv(108 // factor, 216 // factor)
        self.down2 = down(216 // factor, 216 // factor)
        self.dbconv4 = res_conv(54 // factor, 108 // factor)
        self.dbconv5 = res_conv(108 // factor, 216 // factor)
        self.dbconv6 = res_conv(216 // factor, 432 // factor)
        self.down3 = down(432 // factor, 432 // factor)
        self.down4 = down(864 // factor, 432 // factor)
        self.dbup1 = res_conv(432 // factor, 216 // factor)
        self.dbup2 = res_conv(216 // factor, 108 // factor)
        self.dbup3 = res_conv(108 // factor, 54 // factor)
        self.dbup4 = res_conv(54 // factor, 54 // factor)
        self.up1 = up(864 // factor, 216 // factor)
        self.dbup5 = res_conv(216 // factor, 108 // factor)
        self.dbup6 = res_conv(108 // factor, 54 // factor)
        self.dbup7 = res_conv(54 // factor, 54 // factor)
        self.up2 = up(432 // factor, 108 // factor)
        self.dbup8 = res_conv(108 // factor, 54 // factor)
        self.dbup9 = res_conv(54 // factor, 54 // factor)
        self.up3 = up(216 // factor, 54 // factor)
        self.dbup10 = res_conv(54 // factor, 54 // factor)
        self.up4 = up(108 // factor, 54 // factor)
        self.outc1 = outconv(54 // factor, n_classes)
        self.outc2 = outconv(54 // factor, n_classes)
        self.outc3 = outconv(54 // factor, n_classes)
        self.outc4 = outconv(54 // factor, n_classes)
        self.outc = outconv(54 // factor, n_classes)
        self.pool = nn.AvgPool3d(2)
        # have to make a second Decoder
        # elf.unpool = nn.ConvTranspose3d(kernel_size=2, stride=2, bias=False)
        # this is what was used in this setting, has to be modified...
        # previously it was this...
        self.unpool = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.unpool1 = nn.ConvTranspose3d(n_classes, n_classes, kernel_size=2, stride=2)
        self.unpool2 = nn.ConvTranspose3d(n_classes,n_classes,  kernel_size=4, stride=4)
        self.unpool3 = nn.ConvTranspose3d(n_classes,n_classes,  kernel_size=8, stride=8)
        self.unpool4 = nn.ConvTranspose3d(n_classes,n_classes,  kernel_size=16, stride=16)
        # self.unpool = nn.Upsample(scale_factor=2, mode='nearest')
        # self.att = res_conv(64,1)
        # self.gapool = nn.AvgPool2d(kernel_size=224)

    def forward(self, x):

        x1 = self.inc(x)
        y1 = self.pool(x)
        z1 = self.inc(y1)
        x2 = self.down1(x1, z1)
        y2 = self.pool(y1)
        z2 = self.inc(y2)
        a1 = self.dbconv1(z2)
        x3 = self.down2(x2, a1)
        y3 = self.pool(y2)
        z3 = self.inc(y3)
        a2 = self.dbconv2(z3)
        a3 = self.dbconv3(a2)
        x4 = self.down3(x3, a3)
        y4 = self.pool(y3)
        z4 = self.inc(y4)
        a4 = self.dbconv4(z4)
        a5 = self.dbconv5(a4)
        a6 = self.dbconv6(a5)
        x5 = self.down4(x4, a6)
        o1 = self.dbup1(x5)
        o1 = self.dbup2(o1)
        o1 = self.dbup3(o1)
        o1 = self.dbup4(o1)
        out1 = self.outc1(o1)
        x6 = self.up1(x5, x4)
        o2 = self.dbup5(x6)
        o2 = self.dbup6(o2)
        o2 = self.dbup7(o2)
        out2 = self.outc2(o2)
        x7 = self.up2(x6, x3)
        o3 = self.dbup8(x7)
        o3 = self.dbup9(o3)
        out3 = self.outc3(o3)
        x8 = self.up3(x7, x2)
        o4 = self.dbup10(x8)
        out4 = self.outc4(o4)
        o5 = self.up4(x8, x1)
        out5 = self.outc(o5)

        # have to modify this unpooling...
        o1 = self.unpool(self.unpool(self.unpool(self.unpool(o1))))
        o2 = self.unpool(self.unpool(self.unpool(o2)))
        o3 = self.unpool(self.unpool(o3))
        o4 = self.unpool(o4)

        # out1 = self.unpool(self.unpool(self.unpool(self.unpool(out1))))
        # out2 = self.unpool(self.unpool(self.unpool(out2)))
        # out3 = self.unpool(self.unpool(out3))
        # out4 = self.unpool(out4)

        out1 = self.unpool4(out1)
        out2 = self.unpool3(out2)
        out3 = self.unpool2(out3)
        out4 = self.unpool1(out4)

        # out = w3*out3 + w4*out4 + w5*out5

        return out1, out2, out3, out4, out5
