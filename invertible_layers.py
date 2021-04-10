import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn

import numpy as np
import pdb

from layers import * 
from utils import * 

# ------------------------------------------------------------------------------
# Abstract Classes to define common interface for invertible functions
# ------------------------------------------------------------------------------

# Abstract Class for bijective functions
class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()

    def forward_(self, x, objective):
        raise NotImplementedError

    def reverse_(self, y, objective):
        raise NotImplementedError

# Wrapper for stacking multiple layers 
class LayerList(Layer):
    def __init__(self, list_of_layers=None):
        super(LayerList, self).__init__()
        self.layers = nn.ModuleList(list_of_layers)

    def __getitem__(self, i):
        return self.layers[i]

    def forward_(self, x, objective):
        for layer in self.layers:
            x, objective = layer.forward_(x, objective)
        return x, objective

    def reverse_(self, x, objective):
        for layer in reversed(self.layers): 
            x, objective = layer.reverse_(x, objective)
        return x, objective


# ------------------------------------------------------------------------------
# Permutation Layers 
# ------------------------------------------------------------------------------

# Shuffling on the channel axis
class Shuffle(Layer):
    def __init__(self, num_channels):
        super(Shuffle, self).__init__()
        indices = np.arange(num_channels)
        np.random.shuffle(indices)
        rev_indices = np.zeros_like(indices)
        for i in range(num_channels): 
            rev_indices[indices[i]] = i

        indices = torch.from_numpy(indices).long()
        rev_indices = torch.from_numpy(rev_indices).long()
        self.register_buffer('indices', indices)
        self.register_buffer('rev_indices', rev_indices)
        # self.indices, self.rev_indices = indices.cuda(), rev_indices.cuda()

    def forward_(self, x, objective):
        return x[:, self.indices], objective

    def reverse_(self, x, objective):
        return x[:, self.rev_indices], objective
        
# Reversing on the channel axis
class Reverse(Shuffle):
    def __init__(self, num_channels):
        super(Reverse, self).__init__(num_channels)
        indices = np.copy(np.arange(num_channels)[::-1])
        indices = torch.from_numpy(indices).long()
        self.indices.copy_(indices)
        self.rev_indices.copy_(indices)

# Invertible 1x1 convolution
class Invertible1x1Conv(Layer, nn.Conv2d):
    def __init__(self, num_channels):
        self.num_channels = num_channels
        nn.Conv2d.__init__(self, num_channels, num_channels, 1, bias=False)

    def reset_parameters(self):
        # initialization done with rotation matrix
        w_init = np.linalg.qr(np.random.randn(self.num_channels, self.num_channels))[0]
        w_init = torch.from_numpy(w_init.astype('float32'))
        w_init = w_init.unsqueeze(-1).unsqueeze(-1)
        self.weight.data.copy_(w_init)

    def forward_(self, x, objective):
        dlogdet = torch.det(self.weight.squeeze()).abs().log() * x.size(-2) * x.size(-1) 
        objective += dlogdet
        output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, \
            self.dilation, self.groups)
 
        return output, objective

    def reverse_(self, x, objective):
        dlogdet = torch.det(self.weight.squeeze()).abs().log() * x.size(-2) * x.size(-1) 
        objective -= dlogdet
        weight_inv = torch.inverse(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
        output = F.conv2d(x, weight_inv, self.bias, self.stride, self.padding, \
                    self.dilation, self.groups)
        return output, objective


# ------------------------------------------------------------------------------
# Layers involving squeeze operations defined in RealNVP / Glow. 
# ------------------------------------------------------------------------------

# Trades space for depth and vice versa
class Squeeze(Layer):
    def __init__(self, input_shape, factor=2):
        super(Squeeze, self).__init__()
        assert factor > 1 and isinstance(factor, int), 'no point of using this if factor <= 1'
        self.factor = factor
        self.input_shape = input_shape

    def squeeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert h % self.factor == 0 and w % self.factor == 0, pdb.set_trace()
        
        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c, h // self.factor, self.factor, w // self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(bs, c * self.factor * self.factor, h // self.factor, w // self.factor)

        return x
 
    def unsqueeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert c >= 4 and c % 4 == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c // self.factor ** 2, self.factor, self.factor, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(bs, c // self.factor ** 2, h * self.factor, w * self.factor)
        return x
    
    def forward_(self, x, objective):
        if len(x.size()) != 4: 
            raise NotImplementedError # Maybe ValueError would be more appropriate

        return self.squeeze_bchw(x), objective
        
    def reverse_(self, x, objective):
        if len(x.size()) != 4: 
            raise NotImplementedError

        return self.unsqueeze_bchw(x), objective


# ------------------------------------------------------------------------------
# Layers involving prior
# ------------------------------------------------------------------------------

# Split Layer for multi-scale architecture. Factor of 2 hardcoded.
class Split(Layer):
    def __init__(self, input_shape):
        super(Split, self).__init__()
        bs, c, h, w = input_shape
        self.conv_zero = Conv2dZeroInit(c // 2, c, 3, padding=(3 - 1) // 2)

    def split2d_prior(self, x):
        h = self.conv_zero(x)
        mean, logs = h[:, 0::2], h[:, 1::2]
        return gaussian_diag(mean, logs)

    def forward_(self, x, objective):
        bs, c, h, w = x.size()
        z1, z2 = torch.chunk(x, 2, dim=1)
        pz = self.split2d_prior(z1)
        self.sample = z2
        objective += pz.logp(z2) 
        return z1, objective

    def reverse_(self, x, objective, use_stored_sample=False):
        pz = self.split2d_prior(x)
        z2 = self.sample if use_stored_sample else pz.sample() 
        z = torch.cat([x, z2], dim=1)
        objective -= pz.logp(z2) 
        return z, objective

# Gaussian Prior that's compatible with the Layer framework
class GaussianPrior(Layer):
    def __init__(self, input_shape, args):
        super(GaussianPrior, self).__init__()
        self.input_shape = input_shape
        if args.learntop: 
            self.conv = Conv2dZeroInit(2 * input_shape[1], 2 * input_shape[1], 3, padding=(3 - 1) // 2)
        else: 
            self.conv = None

    def forward_(self, x, objective):
        mean_and_logsd = torch.cat([torch.zeros_like(x) for _ in range(2)], dim=1)
        
        if self.conv: 
            mean_and_logsd = self.conv(mean_and_logsd)

        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)

        pz = gaussian_diag(mean, logsd)
        objective += pz.logp(x) 

        # this way, you can encode and decode back the same image. 
        return x, objective

    def reverse_(self, x, objective):
        bs, c, h, w = self.input_shape
        mean_and_logsd = torch.cuda.FloatTensor(bs, 2 * c, h, w).fill_(0.)
        
        if self.conv: 
            mean_and_logsd = self.conv(mean_and_logsd)

        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)
        pz = gaussian_diag(mean, logsd)
        z = pz.sample() if x is None else x
        objective -= pz.logp(z)

        # this way, you can encode and decode back the same image. 
        return z, objective
         

# ------------------------------------------------------------------------------
# Coupling Layers
# ------------------------------------------------------------------------------

# Additive Coupling Layer
class AdditiveCoupling(Layer):
    def __init__(self, num_features):
        super(AdditiveCoupling, self).__init__()
        # assert num_features % 2 == 0
        self.NN = NN(math.ceil(num_features / 2), channels_out=math.floor(num_features / 2))

    def forward_(self, x, objective):
        z1, z2 = torch.chunk(x, 2, dim=1)
        z2 += self.NN(z1)
        return torch.cat([z1, z2], dim=1), objective

    def reverse_(self, x, objective):
        z1, z2 = torch.chunk(x, 2, dim=1)
        z2 -= self.NN(z1)
        return torch.cat([z1, z2], dim=1), objective

# Additive Coupling Layer
class AffineCoupling(Layer):
    def __init__(self, num_features):
        super(AffineCoupling, self).__init__()
        # assert num_features % 2 == 0
        self.NN = NN(math.ceil(num_features/2), channels_out=math.floor(num_features/2)*2)

    def forward_(self, x, objective):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = F.sigmoid(h[:, 1::2] + 2.)
        z2 += shift
        z2 *= scale
        objective += flatten_sum(torch.log(scale))

        return torch.cat([z1, z2], dim=1), objective

    def reverse_(self, x, objective):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = F.sigmoid(h[:, 1::2] + 2.)
        z2 /= scale
        z2 -= shift
        objective -= flatten_sum(torch.log(scale))
        return torch.cat([z1, z2], dim=1), objective


# ------------------------------------------------------------------------------
# Normalizing Layers
# ------------------------------------------------------------------------------

# ActNorm Layer with data-dependant init
class ActNorm(Layer):
    def __init__(self, num_features, logscale_factor=1., scale=1.):
        super(Layer, self).__init__()
        self.initialized = False
        self.logscale_factor = logscale_factor
        self.scale = scale
        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features, 1)))
        self.register_parameter('logs', nn.Parameter(torch.zeros(1, num_features, 1)))

    def forward_(self, input, objective):
        input_shape = input.size()
        input = input.view(input_shape[0], input_shape[1], -1)

        if not self.initialized: 
            self.initialized = True
            unsqueeze = lambda x: x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = input.size(0) * input.size(-1)
            b = -torch.sum(input, dim=(0, -1)) / sum_size
            vars = unsqueeze(torch.sum((input + unsqueeze(b)) ** 2, dim=(0, -1))/sum_size)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6)) / self.logscale_factor
          
            self.b.data.copy_(unsqueeze(b).data)
            self.logs.data.copy_(logs.data)

        logs = self.logs * self.logscale_factor
        b = self.b
        
        output = (input + b) * torch.exp(logs)
        dlogdet = torch.sum(logs) * input.size(-1) # c x h  

        return output.view(input_shape), objective + dlogdet

    def reverse_(self, input, objective):
        assert self.initialized
        input_shape = input.size()
        input = input.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        output = input * torch.exp(-logs) - b
        dlogdet = torch.sum(logs) * input.size(-1) # c x h  

        return output.view(input_shape), objective - dlogdet

# (Note: a BatchNorm layer can be found in previous commits)


# ------------------------------------------------------------------------------
# Stacked Layers
# ------------------------------------------------------------------------------

# 1 step of the flow (see Figure 2 a) in the original paper)
class RevNetStep(LayerList):
    def __init__(self, num_channels, args):
        super(RevNetStep, self).__init__()
        self.args = args
        layers = []
        if args.norm == 'actnorm': 
            layers += [ActNorm(num_channels)]
        else: 
            assert not args.norm               
 
        if args.permutation == 'reverse':
            layers += [Reverse(num_channels)]
        elif args.permutation == 'shuffle': 
            layers += [Shuffle(num_channels)]
        elif args.permutation == 'conv':
            layers += [Invertible1x1Conv(num_channels)]
        else: 
            raise ValueError

        if args.coupling == 'additive': 
            layers += [AdditiveCoupling(num_channels)]
        elif args.coupling == 'affine':
            layers += [AffineCoupling(num_channels)]
        else: 
            raise ValueError

        self.layers = nn.ModuleList(layers)


# Full model
class Glow_(LayerList, nn.Module):
    def __init__(self, input_shape, args):
        super(Glow_, self).__init__()
        layers = []
        output_shapes = []
        _, C, H, W = input_shape
        
        for i in range(args.n_levels):
            # Squeeze Layer 
            layers += [Squeeze(input_shape)]
            C, H, W = C * 4, H // 2, W // 2
            output_shapes += [(-1, C, H, W)]
            
            # RevNet Block
            layers += [RevNetStep(C, args) for _ in range(args.depth)]
            output_shapes += [(-1, C, H, W) for _ in range(args.depth)]

            if i < args.n_levels - 1: 
                # Split Layer
                layers += [Split(output_shapes[-1])]
                C = C // 2
                output_shapes += [(-1, C, H, W)]

        layers += [GaussianPrior((args.batch_size, C, H, W), args)]
        output_shapes += [output_shapes[-1]]
        
        self.layers = nn.ModuleList(layers)
        self.output_shapes = output_shapes
        self.args = args
        self.flatten()

    def forward(self, *inputs):
        return self.forward_(*inputs)

    def sample(self):
        with torch.no_grad():
            samples = self.reverse_(None, 0.)[0]
            return samples

    def flatten(self):
        # flattens the list of layers to avoid recursive call every time. 
        processed_layers = []
        to_be_processed = [self]
        while len(to_be_processed) > 0:
            current = to_be_processed.pop(0)
            if isinstance(current, LayerList):
                to_be_processed = [x for x in current.layers] + to_be_processed
            elif isinstance(current, Layer):
                processed_layers += [current]
        
        self.layers = nn.ModuleList(processed_layers)

    
