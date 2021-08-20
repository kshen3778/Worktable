from .parts import *

################################################################################
############### Modified Deepmind Model ########################################
################################################################################

'''
Aug 21 - changed all (1,3,3) to (3,3,3)
       - changed input from (1, 11,256,256) to (1, 15,256,256)
       - added
'''

class simpleUNet(nn.Module):
    def __init__(self, num_filters=32, num_classes=4, mode='default', debug=False):
        '''
        :param num_filters: Number of initial filters in the model...
        :param num_classes: Number of output classes
        :param mode: Will we sum or crop skip connections in deconv layer(s)
        :param sub_enc: Will we add a residual 1x1x1 sub encoder in the encoding blocks?
        '''
        super().__init__()
        # modified for skip connections before max-pool
        self.debug = debug
        self.classes = num_classes
        self.conv1 = EncoderBlock1 (1,16, pool=False)
        self.conv2 = EncoderBlock1 (16, 32)
        self.conv3 = EncoderBlock1 (32, 64)
        self.conv4 = EncoderBlock1 (64, 128)
        self.conv5 = EncoderBlock2 (128, 256)
        self.conv6 = EncoderBlock2 (256, 512)
        self.conv7 = EncoderBlock3 (512, 512)
        self.conv8 = nn.Conv3d (512, 1024, kernel_size= (1, 4, 4), bias=False)
        # self.fc = nn.Linear ()
        self.fc1 = nn.Sequential(
            nn.LeakyReLU(0.05),
            nn.Linear(1024, 1024, bias=False),
            nn.Dropout(0.5),
        )

        self.fc2 = nn.Sequential(
            nn.LeakyReLU (0.05),
            nn.Linear (1024, 512*4*4, bias=False),
            nn.Dropout(0.5),
        )

        self.upconv0 = DecoderBlock1 (512, 256, mode=mode)
        self.upconv1 = DecoderBlock1 (256, 128, mode=mode)
        self.upconv2 = DecoderBlock1 (128, 64, mode=mode)
        self.upconv3 = DecoderBlock1 (64, 32, mode=mode)
        self.upconv4 = DecoderBlock1 (32, 16, mode=mode)
        self.upconv5 = DecoderBlock1 (16, num_classes, mode=mode, last=True)
        self.relu = nn.LeakyReLU(0.1)
        self.mode = mode

    def forward(self, x):
        print(x.size()) if self.debug is True else None
        conv1 = self.conv1(x) # torch.Size([B, 32, 11, 128, 128])
        print("conv1:", conv1.size()) if self.debug is True else None
        conv2 = self.conv2(conv1) # torch.Size([B, 64, 11, 64, 64])
        print('conv2:', conv2.size()) if self.debug is True else None
        conv3 = self.conv3(conv2) # torch.Size([B, 128, 9, 32, 32])
        print(conv3.size()) if self.debug is True else None
        conv4 = self.conv4(conv3) # torch.Size([B, 128, 7, 16, 16])
        print(conv4.size()) if self.debug is True else None
        conv5 = self.conv5(conv4) # torch.Size([B, 256, 1, 16, 16]
        batch_size = conv5.size() [0]
        print(conv5.size()) if self.debug is True else None
        conv6 = self.conv6(conv5)
        print ("conv6:", conv6.size()) if self.debug is True else None
        conv7 = self.conv7(conv6)
        print ("conv7:", conv7.size()) if self.debug is True else None
        conv8 = self.conv8(conv7)
        # feed forward fully connected layer(s)
        # full1 = self.fc1 (conv7.reshape(batch_size, 512*4*4))
        full1 = self.fc1 (conv8.reshape(batch_size, 1024))
        print("full1:", full1.size()) if self.debug is True else None
        # full1 = torch.add (full1, conv6.reshape(batch_size, 512*4*4))
        # print(full1.size()) if self.debug is True else None
        # copy the full2 vector, does mess up backprop if not copied...
        # full2_copy = full2.clone() if self.to_classify is True else None
        print(full2.size()) if self.debug is True else None
        # without add just full1
        full3 = self.fc2 (torch.add(full1,conv8.reshape(batch_size, 1024))).reshape(batch_size,512,1,4,4)
        # full4 = torch.add (full3, full2).reshape(batch_size,512,1,4,4)
        # print(full4.size()) if self.debug is True else None

        x = self.upconv0(torch.add(conv7, full3), conv6)
        x = self.upconv1(x, conv5)
        print("x:", x.size(), "conv5:", conv5.size()) if self.debug is True else None
        x = self.upconv2(x, conv4)
        print("x:", x.size(), "conv4:", conv4.size()) if self.debug is True else None
        x = self.upconv3(x, conv3)
        print("x:", x.size(), "conv3:", conv3.size()) if self.debug is True else None
        x = self.upconv4(x, conv2)
        print("x:", x.size(), "conv2:", conv2.size()) if self.debug is True else None
        x = self.upconv5(x, conv1)
        print("x:", x.size(), "conv1:", conv1.size()) if self.debug is True else None

        print ("final x:", x.size()) if self.debug is True else None

        return x  # torch.sigmoid(x)
        # pass x, do the sigmoid afterwards
