import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
import pdb

'''
Convolution Layer with zero initialisation
'''
class Conv2dZeroInit(nn.Conv2d):
    def reset_parameters(self):
        self.weight.data.fill_(0.)
        self.bias.data.fill_(0.)

'''
Convolution Interlaced with Actnorm
'''
class Conv2dActNorm(nn.Module):
    def __init__(self, channels_in, channels_out, filter_size, stride=1, padding=None):
        from invertible_layers import ActNorm
        super(Conv2dActNorm, self).__init__()
        padding = (filter_size - 1) // 2 or padding
        self.conv = nn.Conv2d(channels_in, channels_out, filter_size, padding=padding)
        self.actnorm = ActNorm(channels_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.actnorm.forward_and_jacobian(x, 0.)[0]
        return x

'''
Linear layer zero initialization
'''
class LinearZeroInit(nn.Linear):
    def reset_parameters(self):
        self.weight.data.fill_(0.)
        self.bias.data.fill_(0.)

'''
Shallow NN used for skip connection. Labelled `f` in the original repo.
'''
class NN(nn.Module):
    def __init__(self, channels_in, channels_out=None, conv_op=nn.Conv2d): #Conv2dActNorm):
        super(NN, self).__init__()
        channels_out = channels_out or channels_in
        self.main = nn.Sequential(*[
            conv_op(channels_in, channels_in, 3, stride=1, padding=(3 - 1) // 2),
            nn.ReLU(True), 
            conv_op(channels_in, channels_in, 1, stride=1, padding=(1 - 1) // 2),
            nn.ReLU(True), 
            Conv2dZeroInit(channels_in, channels_out, 3, stride=1, padding=(3 - 1) // 2)])

    def forward(self, x):
        return self.main(x)

