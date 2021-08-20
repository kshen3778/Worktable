import torch as t
import torch.nn as nn

class FocalDiceLoss(nn.Module):
    '''
    Link: https://github.com/EKami/carvana-challenge/blob/original_unet/src/nn/losses.py
    '''
    def __init__(self, weight=None):
        super(FocalDiceLoss, self).__init__()
        self.weight = weight

    def forward(self, logits, targets):
        alpha = 0.5
        beta  = 0.5
        logits = logits.cpu()
        ones = t.ones_like(logits).type_as(logits)
        # K.ones(K.shape(y_true))
        # print(type(ones.data), type(y_true.data), type(y_pred.data), ones.size(), y_pred.size())
        p0 = logits #.requires_grad_(True) # proba that voxels are class i
        p1 = (ones-logits) # .requires_grad_(True) # proba that voxels are not class i
        g0 = targets #.requires_grad_(True) #.float()
        g1 = (ones-g0) #.requires_grad_(True)
        num = t.sum(t.sum(t.sum(t.sum(p0*g0*t.pow(1-p0,2), 4),3),2),0) #(0,2,3,4)) #K.sum(p0*g0, (0,1,2,3))
        den = num + alpha*t.sum(t.sum(t.sum(t.sum(p0*g1,4),3),2),0) + beta*t.sum(t.sum(t.sum(t.sum(p1*g0,4),3),2),0) #(0,2,3,4))

        T = t.sum(((num * self.weight)/(den+1e-6)))# * t.pow(1-num/(t.sum(t.sum(t.sum(t.sum(g0,4),3),2),0)+1e-5),2))
        #     Ncl = y_pred.size(1)*1.0
        #     print(Ncl, T)
        return 1.6 * (t.sum(self.weight)- T).requires_grad_(True) #Ncl-T
