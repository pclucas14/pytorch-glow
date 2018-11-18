import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from torch.nn.modules.batchnorm import _BatchNorm

import numpy as np
import pdb
import os

# ------------------------------------------------------------------------------
# Utility Methods
# ------------------------------------------------------------------------------

def flatten_sum(logps):
    while len(logps.size()) > 1: 
        logps = logps.sum(dim=-1)
    return logps

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

def save_session(model, optim, args, epoch):
    path = os.path.join(args.save_dir, str(epoch))
    if not os.path.exists(path):
        os.makedirs(path)

    # save the model and optimizer state
    torch.save(model.state_dict(), os.path.join(path, 'model.pth'))
    torch.save(optim.state_dict(), os.path.join(path, 'optim.pth'))
    print('Successfully saved model')

def load_session(model, optim, args):
    try: 
        start_epoch = int(args.load_dir.split('/')[-1])
        model.load_state_dict(torch.load(os.path.join(args.load_dir, 'model.pth')))
        optim.load_state_dict(torch.load(os.path.join(args.load_dir, 'optim.pth')))
        print('Successfully loaded model')
    except Exception as e:
        pdb.set_trace()
        print('Could not restore session properly')

    return model, optim, start_epoch


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
            eps = torch.zeros_like(mean).normal_()
            return mean + torch.exp(logsd) * eps

    o.logp    = lambda x: flatten_sum(o.logps(x))
    return o


