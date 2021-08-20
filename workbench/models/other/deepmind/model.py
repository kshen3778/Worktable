from .parts import *

################################################################################
############### Modified Deepmind Model ########################################
################################################################################

'''
Aug 21 - changed all (1,3,3) to (3,3,3)
       - changed input from (1, 11,256,256) to (1, 15,256,256)
       - added
'''

class DeepUNet(nn.Module):
    def __init__(self, num_filters=32, num_classes=4, sub_enc=False, debug=False):
        '''
        :param num_filters: Number of initial filters in the model...
        :param num_classes: Number of output classes
        :param mode: Will we sum or crop skip connections in deconv layer(s)
        :param sub_enc: Will we add a residual 1x1x1 sub encoder in the encoding blocks?
        '''

        super().__init__()
        self.debug = debug
        self.classes = num_classes
        self.conv1 = EncoderBlock1(1, num_filters, pool=False, sub_enc=sub_enc)
        self.conv2 = EncoderBlock1(num_filters, 64, sub_enc=sub_enc)
        self.conv3 = EncoderBlock1 (64, 128, sub_enc=sub_enc)
        self.conv4 = EncoderBlock2 (128, 128, sub_enc=sub_enc)
        self.conv5 = EncoderBlock2 (128, 256, sub_enc=sub_enc)
        self.conv6 = EncoderBlock3 (256, 256, sub_enc=sub_enc)
        self.conv7 = nn.Conv3d (256, 1024, kernel_size= (1, 6, 6), bias=False)
        #  self.fc = nn.Linear ()
        self.fc1 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, 1024, bias=False),
            nn.Dropout(0.5),
            #  nn.Linear(in_ // reduction, in_, bias=False),
            #  nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.LeakyReLU (0.1, inplace=True),
            nn.Linear (1024, 6*6*256, bias=False),
            nn.Dropout(0.5),
            #  nn.Linear(in_ // reduction, in_, bias=False),
            #  nn.Sigmoid()
        )
        self.upconv1 = DecoderBlock1 (256, 256, up_scale=False, sub_enc=sub_enc)
        self.upconv2 = DecoderBlock1 (256, 128, sub_enc=sub_enc)
        self.upconv3 = DecoderBlock1 (128, 128, sub_enc=sub_enc)
        self.upconv4 = DecoderBlock1 (128, 64, sub_enc=sub_enc)
        self.upconv5 = DecoderBlock1 (64, num_filters, sub_enc=sub_enc)
        self.upconv6 = DecoderBlock1 (num_filters, num_classes, sub_enc=sub_enc, last=True)
        self.relu = nn.LeakyReLU (0.5)

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
        conv5 = self.conv5(conv4) # torch.Size([B, 256, 1, 16, 16])
        print(conv5.size()) if self.debug is True else None
        conv6 = self.conv6(conv5) # torch.Size([B, 256, 1, 8, 8])
        batch_size = conv6.size() [0]
        print(conv6.size()) if self.debug is True else None
        conv7 = self.conv7(conv6)
        print ("conv7:", conv7.size()) if self.debug is True else None
        # feed forward fully connected layer(s)
        full1 = self.fc1 (conv7.reshape(batch_size,1024))
        print("full1:", full1.size()) if self.debug is True else None
        full1 = torch.add (full1, conv7.reshape(batch_size,1024))
        print(full1.size()) if self.debug is True else None
        full2 = torch.add (self.fc1 (full1), full1)
        # copy the full2 vector, does mess up backprop if not copied...
        # full2_copy = full2.clone() if self.to_classify is True else None
        print(full2.size()) if self.debug is True else None
        full3 = self.relu(self.fc2 (full2)).reshape(batch_size, 256,1,6,6)
        print(full3.size()) if self.debug is True else None
        x = self.upconv1(full3, conv6)
        print("x:", x.size(), "conv5:", conv5.size()) if self.debug is True else None
        x = self.upconv2(x, conv5)
        print("x:", x.size(), "conv4:", conv4.size()) if self.debug is True else None
        x = self.upconv3(x, conv4)
        print("x:", x.size(), "conv3:", conv3.size()) if self.debug is True else None
        x = self.upconv4(x, conv3)
        print("x:", x.size(), "conv2:", conv2.size()) if self.debug is True else None
        x = self.upconv5(x, conv2)
        print("x:", x.size(), "conv1:", conv1.size()) if self.debug is True else None
        x = self.upconv6(x, conv1)
        print ("final x:", x.size()) if self.debug is True else None
        # only true if z padding is implemented in deconv block(s)

        return x

        # then pass to Dice for overlap metric...
        # this was for patient classification
        # if you do to much most likely overkill...
        # if self.to_classify is True:
        #     full2_copy = self.classifier(full2_copy)
        #     # check if we have to do this
        #     return torch.sigmoid(x), full2_copy
        #     # pass labels into CCE loss with class label without softmax
        #
        # else:
