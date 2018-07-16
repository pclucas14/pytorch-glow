import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from torch.nn.modules.batchnorm import _BatchNorm

import numpy as np
import pdb

'''
Utilities
'''
def flatten_sum(logps):
    while len(logps.size()) > 1: 
        logps = logps.sum(dim=-1)
    return logps

'''
Distributions
'''
def standard_gaussian(shape):
    mean, logsd = [torch.cuda.FloatTensor(shape).fill_(0.) for _ in range(2)]
    return gaussian_diag(mean, logsd)

def gaussian_diag(mean, logsd):
    class o(object):
        pass

        @property
        def eps(self):
            return torch.cuda.FloatTensor(mean.size()).normal_()
            
    o.mean    = mean
    o.logsd   = logsd
    # o.eps   = tf.random_normal(tf.shape(mean))
    o.sample  = mean + torch.exp(logsd) * o.eps
    o.sample2 = lambda eps: mean + torch.exp(logsd) * eps
    o.logp    = lambda x: flatten_sum(o.logps(x))
    o.get_eps = lambda x: (x - mean) / tf.exp(logsd)
    o.logps   = lambda x: -0.5 * (np.log(2 * np.pi).astype('float32') \ 
                          + 2. * logsd + (x - mean) ** 2 / torch.exp(2. * logsd))
    return o


def discretized_logistic(mean, logscale, binsize=1. / 256):
    class o(object):
        pass
    o.mean = mean
    o.logscale = logscale
    scale = tf.exp(logscale)

    def logps(x):
        x = (x - mean) / scale
        return torch.log(F.sigmoid(x + binsize / scale) - F.sigmoid(x) + 1e-7)
    o.logps = logps
    o.logp = lambda x: flatten_sum(logps(x))
    return o
