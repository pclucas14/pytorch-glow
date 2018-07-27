import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn

import numpy as np
import pdb

'''
Convolution Layer with zero initialisation
'''
class Conv2dZeroInit(nn.Conv2d):
    def __init__(self, channels_in, channels_out, filter_size, stride=1, padding=0, logscale=3.):
        super().__init__(channels_in, channels_out, filter_size, stride=stride, padding=padding)
        self.register_parameter("logs", nn.Parameter(torch.zeros(channels_out, 1, 1)))
        self.logscale_factor = logscale

    def reset_parameters(self):
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        out = super().forward(input)
        return out * torch.exp(self.logs * self.logscale_factor)

'''
Convolution Interlaced with Actnorm
'''
class Conv2dActNorm(nn.Module):
    def __init__(self, channels_in, channels_out, filter_size, stride=1, padding=None):
        from invertible_layers import ActNorm
        super(Conv2dActNorm, self).__init__()
        padding = (filter_size - 1) // 2 or padding
        self.conv = nn.Conv2d(channels_in, channels_out, filter_size, padding=padding, bias=False)
        self.actnorm = ActNorm(channels_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.actnorm.forward_(x, -1)[0]
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
def NN(in_channels, hidden_channels=512, channels_out=None):
    channels_out = channels_out or in_channels
    return nn.Sequential(
        Conv2dActNorm(in_channels, hidden_channels, 3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        Conv2dActNorm(hidden_channels, hidden_channels, 1, stride=1, padding=0),
        nn.ReLU(inplace=True),
        Conv2dZeroInit(hidden_channels, channels_out, 3, stride=1, padding=1))
