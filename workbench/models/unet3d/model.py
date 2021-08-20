from .parts import *

class UNet3D(nn.Module):

    """

    conv => BN => ReLU * 2 ;
    num_classes = LCTV, RCTV, GTV
    Can integrate check on the fly how many classes there are automatically
    # should make 2 versions of this.
    One 2.5 version.
    One 3D version.

    """

    def __init__(self, num_classes=1, num_filters=32,  drop_rate=0.2,
                 scale=1, debug=False, fconcat=False, project=False,
                 deformable=False):

        super().__init__()
        self.debug = debug
        self.classes=num_classes
        self.dropout = nn.Dropout3d(drop_rate)
        self.relu = nn.LeakyReLU(0.1)
        # padded 3x3x3, avg_pool 2x2x2
        # no pooling first block
        # (B, 1, 64, 256, 256)
        self.conv1 = EncoderBlock(1, int(16*scale), pool=False, layers=3) # 64, 200, 200
        # (B, 64, 64, 256, 256 )
        self.conv2 = EncoderBlock(int(16*scale), int(32*scale), layers=3, deformable=deformable) # 32, 100, 100
        self.up1 = UpAddBlock(int(32*scale), int(16*scale), kernel=2, stride=2)
        # (B, 64, 32, 128, 128 )
        self.conv3 = EncoderBlock(int(32*scale), int(64*scale), layers=4, deformable=deformable)  # 16, 50, 50
        self.up2 = UpAddBlock(int(64*scale), int(16*scale), kernel=4, stride=4)
        self.up22 = UpAddBlock(int(64*scale), int(32*scale), kernel=2, stride=2)
        # (B, 128, 16, 64, 64)
        self.conv4 = EncoderBlock(int(64*scale), int(128*scale), layers=4, deformable=deformable) # 8, 25, 25
        self.up3 = UpAddBlock(int(128*scale), int(16*scale), kernel=8, stride=8)
        self.up32 = UpAddBlock(int(128*scale), int(32*scale), kernel=4, stride=4)
        # (B, 256, 8, 32, 32)
        # self.conv5 = nn.Conv3d (128*scale, 128*scale, kernel_size= (1, 1, 1), bias=False) # 8, 32, 32
        # up convolution
        self.cat = fconcat
        project=False
        self.upconv1 = DecoderBlock(int(128*scale), int(64*scale), project=project, conv_lay=2) if fconcat is False else DecoderBlock(256*scale, 128*scale, project=project, conv_lay=3)
        self.up4 = UpAddBlock(int(64*scale), int(16*scale), kernel=4, stride=4)
        self.upconv2 = DecoderBlock(int(64*scale), int(32*scale), project=project, conv_lay=2, deformable=False)
        self.up5 = UpAddBlock(int(32*scale), int(16*scale), kernel=2, stride=2)
        self.upconv3 = DecoderBlock(int(32*scale), num_classes, last=True, project=project)
        # self.upconv4 = DecoderBlock(16*scale, num_classes, last=True, project=project)
        # self.pre_conv = nn.Conv3d(1, num_classes, kernel_size=(1, 1, 1), bias=True)

    def forward(self, x):

        # ensure dim into convnet is 5!!
        # squeeze_back=False
        # try volume with 54 slices
        if x.dim() < 5:
            x = x.unsqueeze(0)
            # squeeze_back = True

        # in_ = x.clone()
        # print(x.size()) if self.debug is True else None
        conv1 = self.conv1(x)
        # print(conv1.size()) if self.debug is True else None
        conv2 = self.conv2(conv1)
        # print(conv2.size()) if self.debug is True else None
        conv3 = self.conv3(conv2)
        # print(conv3.size()) if self.debug is True else None
        conv4 = self.conv4(conv3)

        # mid level skip connections...
        conv1 = self.up1(conv2, conv1)
        conv1 = self.up2(conv3, conv1)
        conv1 = self.up3(conv4, conv1)
        conv2 = self.up22(conv3, conv2)
        conv2 = self.up32(conv4, conv2)

        # print(mid.size()) if self.debug is True else None
        x = self.upconv1(conv4, conv3)
        conv1 = self.up4(x, conv1)
        # print(x.size()) if self.debug is True else None
        x = self.upconv2(x, conv2)
        conv1 = self.up5(x, conv1)
        # print(x.size()) if self.debug is True else None
        # x = torch.add(x, x2)
        x = self.upconv3(x, conv1)

        return x
