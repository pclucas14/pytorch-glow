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

def preprocess(x, args, add_noise=True):
    x = x.float()
    if args.n_bits_x < 8: 
        x = torch.floor(x / 2 ** (8 - args.n_bits_x))
    
    x = x / args.n_bins - .5
    if add_noise: 
        x += torch.FloatTensor(x.size()).uniform_(0., 1./args.n_bins)

    return x

def postprocess(x, args):
    x = (x + .5) * args.n_bins
    x = np.floor(x) * (256./args.n_bins)
    return np.clip(x, 0., 255).astype('uint8')

'''
Distributions
'''
def standard_gaussian(shape):
    mean, logsd = [torch.cuda.FloatTensor(shape).fill_(0.) for _ in range(2)]
    return gaussian_diag(mean, logsd)

def gaussian_diag(mean, logsd):
    class o(object):
        pass

        def logps(x):
            return  -0.5 * (torch.cuda.FloatTensor([np.log(2 * np.pi).astype('float32')]) + 2. * logsd + (x - mean) ** 2 / torch.exp(2. * logsd))

        def sample():
            eps = torch.cuda.FloatTensor(mean.size()).normal_(0,1)
            return mean + torch.exp(logsd) * eps

    o.mean    = mean
    o.logsd   = logsd
    # o.eps   = tf.random_normal(tf.shape(mean))
    o.logp    = lambda x: flatten_sum(o.logps(x))
    o.get_eps = lambda x: (x - mean) / tf.exp(logsd)
    # o.logps   = lambda x: -0.5 * (np.log(2 * np.pi).astype('float32') \
    #                      + 2. * logsd + (x - mean) ** 2 / torch.exp(2. * logsd))
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
