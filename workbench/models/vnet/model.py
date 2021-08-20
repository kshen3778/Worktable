from .parts import *

'''
Pytorch V-Net implementation taken from
https://github.com/mattmacy/vnet.pytorch/blob/master/vnet.py

# uses loss functions from
import torchbiomed.loss as bioloss
# can use that as well...
https://github.com/mattmacy/torchbiomed/blob/master/torchbiomed/loss.py

'''

class VNet3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    # 128x128x64
    # make input 192x192x64
    def __init__(self, num_classes=1, elu=True, nll=False):
        super(VNet3D, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 2, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(128, 64, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(64, 32, 1, elu)
        self.up_tr32 = UpTransition(32, 16, 1, elu)
        # commented out softmax layer
        self.out_tr = OutputTransition(16, num_classes, elu, nll)

    # The network topology as described in the diagram
    # in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    #     # the number of convolutions in each layer corresponds
    #     # to what is in the actual prototxt, not the intent
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        out16 = self.in_tr(x)
        # print('one:', out16.size())
        out32 = self.down_tr32(out16)
        # print('two', out32.size())
        out64 = self.down_tr64(out32)
        # print('three', out64.size())
        out128 = self.down_tr128(out64)
        # print('four:', out128.size())
        out256 = self.down_tr256(out128)
        print('final ', out256.size())
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out , out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)

        return out
