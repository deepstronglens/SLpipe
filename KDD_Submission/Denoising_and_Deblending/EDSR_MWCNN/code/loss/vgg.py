from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import os
class VGG(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(VGG, self).__init__()
        vgg19 = models.vgg19()
        vgg19.load_state_dict(torch.load(os.path.join('/home/sand33p/EDSR-PyTorch/code/loss/', 'vgg19-dcbb9e9d.pth')))
        vgg_features = vgg19.features
        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])

        self.vgg.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x_trnsf = x.repeat(1, 3, 1, 1)
            x = self.vgg(x_trnsf)
            return x
        vgg_sr = _forward(sr)
	    with torch.no_grad():
	            vgg_hr = _forward(hr.detach())
        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss
