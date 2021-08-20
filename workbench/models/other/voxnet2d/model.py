from .parts import *

# ##############################################################################
# ############## DeepVoxNet ###################################################
# instead of 2 Denseblocks >>> 4 Denseblocks (6 layers each?? v  )
################################################################################
################################################################################

class DeepVoxNet(nn.Module):

    def __init__(self, in_filters=16, mid_filters=160, num_classes=1, drop=True):
        super().__init__()
        self.classes=num_classes
        #  input dims : (B=N,C=1,Z=11,W=256,H=256)
        #  The input affects the deepness of the architecture...
        #  make 2.5D DeepVoxNet
        self.conv1 = nn.Sequential(
                            nn.Conv3d(1, in_filters, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(1,1,1)),
                            nn.BatchNorm3d(in_filters),
                            # padding = (0, 1, 1)?
                            # nn.Con3d(in_filters, in_filters, kernel_size=(3, 1, 1)),
                            )

        self.mid_block = nn.Sequential(
                        ConvMe3d(mid_filters, mid_filters, kernel_size=(1, 1, 1), padding=0),
                        nn.BatchNorm3d(mid_filters),
                        )

        self.mid_deconv = nn.Sequential(
                        ConvMe3d(mid_filters, mid_filters, kernel_size=(3,1,1), padding=0),
                        ConvMe3d(mid_filters, mid_filters, kernel_size=(3,1,1), padding=0),
                        ConvMe3d(mid_filters, mid_filters, kernel_size=(3,1,1), padding=0),
                        nn.BatchNorm3d(mid_filters),
                        nn.ConvTranspose3d(mid_filters, num_classes, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                        )

        self.final_deconv = nn.Sequential(

                        ConvMe3d(304, 304, kernel_size=(3, 1, 1), padding=0),
                        nn.BatchNorm3d(304),
                        nn.ConvTranspose3d(304, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                        nn.ConvTranspose3d(128, num_classes, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                        )

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.db1 = DenseBlock(16,  12)
        self.db2 = DenseBlock(88,  12)
        self.db3 = DenseBlock(mid_filters, 12)
        self.db4 = DenseBlock(232, 12)

        # there is [most likely] a better way to structure this...
        # Get to it dumb but!

    def forward(self, x):

        # add a Unet Configuration of this model...
        # variation of this - combine previous layers in UnetLike Skip Cons
        # save it in memory only if we plan to make unet out of this...??
        # can test Regular vs UNET.
        # print(x.size())
        x = self.conv1(x)  # (B, 16, 9, 252, 252)
        # set a pool variable = False or True...
        # might want to make a smaller dense block?
        # print(x.size())
        x = self.db1(x)  # POOL: (B, 76, 7, 126, 126)
        # print(x.size())
        x = self.db2(x)  # POOL: (B, 160, 7, 64, 64)
        # print(x.size())
        x = self.mid_block(x)  # (B, 160, 7, 64, 64)
        # print(x.size())
        # Use this for long skip connection...
        mid_skip = self.mid_deconv(x)
        # print('Mid Conv ', mid_skip.size())
        x = self.pool(x)
        # print('After Mid Pool:', x.size())
        x = self.db3(x)
        # print(x.size())
        x = self.db4(x)
        # print(x.size())
        x = self.final_deconv(x)
        # print('After Final Deconv Layer ', x.size())
        x = torch.cat([mid_skip, x], dim=1)
        # print('After Final Concat ', x.size())
        x = x.sum(dim=1)
        # print('After Final Sum ', x.size())
        # insert first deconvolutional block here...
        # (B, 160, 5, 64, 64)
        # make sure we end with padding...
        # (B, 304, 3, 32, 32)
        # concatenate then sum on axis 1
        # needs them to have the same dimensions...
        # final output should be (B, 1, 1, 256, 256)
        return torch.sigmoid(x)
