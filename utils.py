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

# torch DataParallel is not sending copies of Models properly on GPU :/
class data_parallel(nn.Module):
    def __init__(self, input, device_ids, output_device=0):
        super().__init__()
        self.module = input
        self.device_ids = device_ids
        self.output_device = output_device

    def forward(self, input):
        if not self.device_ids:
            return self.module(input)

        if self.output_device is None:
            self.output_device = device_ids[0]

        pdb.set_trace()

        replicas = nn.parallel.replicate(self.module, self.device_ids)
        inputs = nn.parallel.scatter(input, self.device_ids)
        replicas = replicas[:len(inputs)]
        outputs = nn.parallel.parallel_apply(replicas, inputs)
        return nn.parallel.gather(outputs, self.output_device)
        

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


