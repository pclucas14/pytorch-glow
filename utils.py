import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from torch.nn.modules.batchnorm import _BatchNorm

import numpy as np
import pdb

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def flatten_sum(logps):
    while len(logps.size()) > 1: 
        logps = logps.sum(dim=-1)
    return logps

def preprocess(x):
    raise Exception
    x = x.float()
    
    x = x / 256. # - .5
    return x

def postprocess(x):
    raise Exception
    return x
    # x = (x + .5) * 256.
    # x = x * 256.
    # x = torch.floor(x) * (256./256.)
    # return torch.clamp(x, min=0., max=255).byte()

# ------------------------------------------------------------------------------
# Distributions
# ------------------------------------------------------------------------------

def standard_gaussian(shape):
    mean, logsd = [torch.cuda.FloatTensor(shape).fill_(0.) for _ in range(2)]
    return gaussian_diag(mean, logsd)

def gaussian_diag(mean, logsd):
    class o(object):
        Log2PI = float(np.log(2 * np.pi))
        pass

        def logps(x):
            return  -0.5 * (o.Log2PI + 2. * logsd + ((x - mean) ** 2) / torch.exp(2. * logsd))

        def sample():
            eps = torch.cuda.FloatTensor(mean.size()).normal_()
            return mean + torch.exp(logsd) * eps

    o.logp    = lambda x: flatten_sum(o.logps(x))
    return o


