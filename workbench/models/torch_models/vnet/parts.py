# make a basic 3D unet architecture ... simple 3D vanilla unet ...
from torch import nn
from torch.nn import functional as F
import torch

# idea is to iterate through all our data. Split patients by Disease site
# save .csv image path, with center of mass of GTV...
# then for that slice take window +/- 31 slices? 63 x 256 x 256
# we will pass that through with the mask...

def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
# class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
#
#     def __init__(self):
#         super(ContBatchNorm3d, self).__init__()
#
#     # def _check_input_dim(self, input):
#     #     if input.dim() != 5:
#     #         raise ValueError('expected 5D input (got {}D input)'
#     #                          .format(input.dim()))
#
#         # super(ContBatchNorm3d, self)._check_input_dim(input)
#
#     def forward(self, input):
#         # self._check_input_dim(input)
#         return F.batch_norm(input, self.running_mean, self.running_var,
#                             self.weight, self.bias, True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu, upconv=False, count=None):
        super(LUConv, self).__init__()

        if upconv is True:
            if count==0:
                self.conv1 = nn.Conv3d(nchan*2, nchan, kernel_size=5, padding=2)
            else:
                self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        else:
            self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)

        self.upconv = upconv
        self.relu1 = ELUCons(elu, nchan)
        self.nchan = nchan
        self.bn1 = nn.BatchNorm3d(nchan)# ContBatchNorm3d(nchan)

    def forward(self, x):
        if self.upconv is True:
            print('channels', self.nchan)
            print('input shape', x.size())
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu, upconv=False):
    layers = []
    count=0
    for i in range(depth):
        layers.append(LUConv(nchan, elu, upconv=upconv, count=i))

    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, outChans, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(outChans) # ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, outChans=None, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans if outChans is None else outChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans) # ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d( outChans) # ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        # when addition can turn upconv == false
        self.ops = _make_nConv(outChans, nConvs, elu, upconv=True)

    def forward(self, x, skipx):
        out = self.do1(x)
        out = self.relu1(self.bn1(self.up_conv(out)))
        skipx = self.do2(skipx)
        # original version...
        out = torch.cat((out, skipx), 1)
        # modified version
        # out = torch.add(out, skipx)
        out = self.ops(out)
        out = self.relu2(torch.add(out, skipx))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_classes, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, n_classes, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(n_classes) # ContBatchNorm3d(n_classes)
        self.conv2 = nn.Conv3d(n_classes, n_classes, kernel_size=1)
        self.relu1 = ELUCons(elu, n_classes)

        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        # original
        # out = out.permute(0, 3, 2, 4, 1).contiguous()
        # out = out.view(out.numel() // 2, 2)
        # out = self.softmax(out)

        # flatten
        out = out.permute(0, 1, 3, 2, 4).contiguous()

        # treat channel 0 as the predicted output
        return out
