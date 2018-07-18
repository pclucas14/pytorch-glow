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
    def __init__(self, channels_in, channels_out=None):
        super(NN, self).__init__()
        channels_out = channels_out or channels_in
        #wn = lambda x: x
        self.main = nn.Sequential(*[
            wn(nn.Conv2d(channels_in, channels_in, 3, stride=1, padding=(3 - 1) // 2)),
            nn.ReLU(True), 
            wn(nn.Conv2d(channels_in, channels_in, 1, stride=1, padding=(1 - 1) // 2)),
            nn.ReLU(True), 
            Conv2dZeroInit(channels_in, channels_out, 3, stride=1, padding=(3 - 1) // 2)])

    def forward(self, x):
        return self.main(x)

