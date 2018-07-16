import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from torch.nn.modules.batchnorm import _BatchNorm

import numpy as np
import pdb

from layers import * 

'''
Abstract Class for bijective functions
'''
class Layer():
    def forward_and_jacobian(self, x, sum_log_det_jacobian):
        raise NotImplementedError

    def reverse_and_jacobian(self, y, z):
        raise NotImplementedError

'''
Wrapper for stacking multiple layers 
'''
class LayerList(Layer):
    def __init__(self, list_of_layers=None):
        self.layers = list_of_layers

    def forward_and_jacobian(self, x, sum_log_det_jacobian):
        for layer in self.layers: 
            x, sum_log_det_jacobian = layer.forward_and_jacobian(x, sum_log_det_jacobian)
        return x, sum_log_det_jacobian

    def reverse_and_jacobian(self, x, sum_log_det_jacobian):
        for layer in reversed(self.layers): 
            x, sum_log_det_jacobian = layer.reverse_and_jacobian(x, sum_log_det_jacobian)
        return x, sum_log_det_jacobian

'''
Shuffling on the channel axis
'''
class Shuffle(Layer):
    def __init__(self, num_channels):
        indices = np.arange(num_channels)
        np.random.shuffle(indices)
        rev_indices = np.zeros_like(indices)
        for i in range(num_channels): 
            rev_indices[indices[i]] = i

        indices = torch.from_numpy(indices).long()
        rev_indices = torch.from_numpy(indices).long()
        self.indices, self.rev_indices = indices.cuda(), rev_indices.cuda()

    def forward_and_jacobian(self, x, sum_log_det_jacbian):
        return x[:, self.indices], sum_log_det_jacobian[:, self.indices]

    def reverse_and_jacobian(self, x, sum_log_det_jacobian):
        return x[:, self.rev_indices], sum_log_det_jacobian[:, self.rev_indices]
        

'''
Reversing on the channel axis
'''
class Reverse(Shuffle):
    def __init__(self, num_channels):
        indices = np.copy(np.arange(num_channel)[::-1])
        self.indices = torch.from_numpy(indices).long()
        self.indices = self.indices.cuda()
        self.rev_indices = self.indices

'''
Invertible 1x1 convolution
'''
class Invertible1x1Conv(Layer, nn.Conv2d):
    def __init__(self, num_channels):
        nn.Conv2d.__init__(self, num_channels, num_channels, 1, bias=False)
        self.num_channels = num_channels

    def reset_parameters(self):
        # initialization done with rotation matrix
        w_init = np.linalg.qr(np.random.randn((self.num_channels, self.num_channels)))[0]
        w_init = torch.from_numpy(w_init.astype('float32'))
        w_init = w_init.cuda()
        w_init = w_init.unsqueeze(-1).unsqueeze(-1)
        self.weight.copy_(w_init)

    def forward_and_jacobian(self, x, sum_log_det_jacobian):
        dlogdet = torch.det(self.weight.squeeze()).abs().log() * x.size(-2) * x.size(-1)
        sum_log_det_jacobian += dlogdet
        output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, \
                    self.dilation, self.groups)
        return output, sum_log_det_jacobian

    def reverse_and_jacobian(self, x, sum_log_det_jacobian):
        dlogdet = torch.det(self.weight.squeeze()).abs().log() * x.size(-2) * x.size(-1)
        sum_log_det_jacobian -= dlogdet
        weight_inv = torch.inverse(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
        output = F.conv2d(x, weight_inv, self.bias, self.stride, self.padding, \
                    self.dilation, self.groups)
        return output, sum_log_det_jacobian


'''
Trades space for depth and vice versa
'''
class Squeeze(Layer):
    def __init__(self, factor=2):
        assert factor > 1 and isinstance(factor, int), 'no point of using this if factor <= 1'
        self.factor = factor

    def squeeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert h % self.factor == 0 and w % self.factor == 0
        
        # done as in GLOW repository
        x = x.transpose(3, 1).contiguous()
        x = x.view(-1, h // self.factor, self.factor, w // self.factor, factor, c)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.view(-1, h // self.factor, w // self.factor, factor, c * self.factor ** 2)
        x = x.transpose(3, 1).contiguous()
 
    def unsqueeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert c >= 4 and c % 4 == 0

        # done as in GLOW repository
        x = x.transpose(3, 1).contiguous()
        x = x.view(-1, h, w, int(c / self.factor ** 2), self.factor, self.factor)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.view(-1, int(h * w), int(w * self.factor), int(c / self.factor ** 2))
        return x.transpose(3, 1).contiguous()
    
    def forward_and_jacobian(self, x, sum_log_det_jacobian):
        if len(x.size()) != 4: 
            raise NotImplementedError # Maybe ValueError would be more appropriate

        return squeeze_bchw(self, x), squeeze_bchw(self, sum_log_det_jacobian)
        
    def reverse_and_jacobian(self, x, sum_log_det_jacobian):
        if len(x.size()) != 4: 
            raise NotImplementedError

        return unsqueeze_bchw(self, x), unsqueeze_bchw(self, sum_log_det_jacobian)

'''
Split Layer for multi-scale architecture. Factor of 2 hardcoded.
'''
class Split(Squeeze):
    def forward_and_jacobian(self, x, sum_log_det_jacobian):
        bs, c, h, w = x.size()
        z1, z2 = torch.chunk(x, 2, dim=1)

'''
Additive Coupling Layer
'''
class AdditiveCoupling(Layer, nn.Module):
    def __init__(self, num_features):
        super(AdditiveCoupling, self).__init__()
        assert num_features % 2 == 0
        self.NN = NN(num_features // 2)

    def forward_and_jacobian(self, x, sum_log_det_jacobian):
        z1, z2 = torch.chunk(x, 2, dim=1)
        z2 += self.NN(z1)
        return torch.cat([z1, z2], dim=1), sum_log_det_jacobian

    def reverse_and_jacobian(self, x, sum_log_det_jacobian):
        z1, z2 = torch.chunk(x, 2, dim=1)
        z2 -= self.NN(z1)
        return torch.cat([z1, z2], dim=1), sum_log_det_jacobian

'''
Additive Coupling Layer
'''
class AffineCoupling(Layer, nn.Module):
    def __init__(self, num_features):
        super(AdditiveCoupling, self).__init__()
        assert num_features % 2 == 0
        self.NN = NN(num_features // 2)

    def forward_and_jacobian(self, x, sum_log_det_jacobian):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = F.sigmoid(h[:, 1::2] + 2.)
        z2 += shift
        z2 *= scale
        # TODO: make shapes work
        sum_log_det_jacobian += torch.log(scale)

        return torch.cat([z1, z2], dim=1), sum_log_det_jacobian

    def reverse_and_jacobian(self, x, sum_log_det_jacobian):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = F.sigmoid(h[:, 1::2] + 2.)
        z2 /= scale
        z2 -= shift
        # TODO: make shapes work
        sum_log_det_jacobian -= torch.log(scale)
        return torch.cat([z1, z2], dim=1), sum_log_det_jacobian

'''
ActNorm Layer with data-dependant init
'''
class ActNorm(Layer):
    def __init__(self):
        raise NotImplementedError

    def forward_and_jacobian(self, x, sum_log_det_jacobian):
        pass

    def reverse_and_jacobian(self, x, sum_log_det_jacobian):
        pass

'''
Batch Normalization, with no affine parameters, and log_det
'''
class BatchNorm(Layer, _BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        # this transformation keeps track of running stats, but is not affine. 
        super(BatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=False)

    def forward_and_jacobian(self, input, sum_log_det_jacobian):
        output = F.batch_norm(input, self.running_mean, self.running_var, None, None, 
            self.training, self.momentum, self.eps)
        
        # Since there is no way to fetch the actual std used during batch norm, we 
        # must recalculate it. It **sems** the following (commented) code replicates the 
        # output of batch norm. So let's use the used standard deviation
        input_shape = input.size()
        input = input.view(input_shape[0], input_shape[1], -1)

        # 2) Compute the sum and the square-sum
        sum_size = input.size(0) * input.size(-1)
        input_sum = input.sum(dim=0).sum(dim=-1)
        input_ssum = (input ** 2).sum(dim=0).sum(dim=-1)

        # 3) compute mean and variance
        mean = input_sum / sum_size
        sumvar = input_ssum - input_sum * mean
        var = (sumvar / sum_size).clamp(self.eps)
        inv_std = var ** -0.5

        log_det_jacobian = torch.log(torch.abs(inv_std))
        # TODO: expand log_det_jacobian to have the same shape as the original input
        sum_log_det_jacovian += log_det_jacobian

        return output, log_det_jacobian

    def reverse(self, input, sum_log_det_jacobian):
        assert not self.training, 'reverse pass should only be used for sampling'

        # 1) Resize the input ot (B, C, -1)
        input_shape = input.size()
        input = input.view(input_shape[0], input_shape[1], -1)

        # 2) Compute the sum and the square-sum
        sum_size = input.size(0) * input.size(-1)
        input_sum = input.sum(dim=0).sum(dim=-1)
        input_ssum = (input ** 2).sum(dim=0).sum(dim=-1)

        # 3) compute mean and variance
        mean = input_sum / sum_size
        sumvar = input_ssum - input_sum * mean
        var = (sumvar / sum_size).clamp(1e-5)
        unbias_var = sumvar / (sum_size - 1)
        inv_std = var ** -0.5

        # 4) normalize, but the other way around
        unsqueeze_ft = lambda x: x.unsqueeze(0).unsqueeze(-1).detach()
        output = input * var ** 0.5
        output = output + mean
        
        log_det_jacobian = torch.log(torch.abs(inv_std))
        # TODO: expand log_det_jacobian to have the same shape as the original input
        sum_log_det_jacobian -= log_det_jacobian

        return output.view(input_shape), log_det_jacobian
        
'''
1 step of the flow (see Figure 2 a) in the original paper)
'''
class RevNetStep(LayerList):
    def __init__(self, num_channels, args):
        self.args = args
        layers = []
        if args.norm == 'actnorm': 
            layers += [ActNorm()]
        elif args.norm == 'batchnorm':
            layers += [BatchNorm(num_channels)]
        else: 
            raise ValueError
        
        if args.permutation == 'reverse':
            layers += [Reverse()]
        elif args.permutation == 'shuffle': 
            layers += [Shuffle()]
        elif args.permutation == 'conv':
            layers += [Invertible1x1Conv(num_channels)]
        else: 
            raise ValueError

        if args.coupling == 'additive': 
            layers += [AdditiveCoupling(num_features)]
        elif args.coupling == 'affine':
            layers += [AffineCoupling(num_features)]
        else: 
            raise ValueError

        self.layers = layers 

'''
1 "scale" i.e. stacking of multiple steps. See Figure 2 b) in the original paper
'''
class RevNet(LayerList):
    def __init__(self, input_shape, args):
        bs, c, h, w = input_shape 
        self.layers = [RevNetStep(c) for _ in range(args.depth)]

if __name__ == '__main__':
