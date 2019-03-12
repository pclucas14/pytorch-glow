import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import flatten_sum

# Layers
# -----------------------------------------------------------------------------
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, zero_init, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

        self.init = False
        self.zero_init = zero_init

    def forward(self, x):
        if not self.init and self.zero_init:
            self.weight.data.zero_()
            self.init = True

        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

# Flows 
# -----------------------------------------------------------------------------
class AF(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=7, num_layers=2, use_bn=True):
        super(AF, self).__init__()
        
        # for now only support odd filter sizes
        assert kernel_size % 2 == 1
        assert num_layers >  1

        pad = (kernel_size - 1) // 2

        layers = [MaskedConv2d('A', False, in_c, out_c, kernel_size, 1, pad, bias=False)]

        for i in range(1, num_layers):
            if use_bn: 
                layers += [nn.BatchNorm2d(out_c)]
            
            int_c     = 2 * in_c if (i+1) == num_layers else out_c
            zero_init = True if (i+1) == num_layers else False
            layers += [nn.ReLU(True)]
            layers += [MaskedConv2d('B', zero_init, out_c, int_c, kernel_size, 1, pad, bias=False)]

        self.main = nn.Sequential(*layers)

    
    def forward(self, x, objective):
        # x : (bs,  C, H, W)

        # h:  (bs, 2C, H, W)
        h = self.main(x)

        # s:  (bs,  C, H, W)
        shift, scale = h[:, ::2], F.sigmoid(h[:, 1::2] + 2.)
        
        x = x + shift
        x = x * scale

        objective += torch.log(scale) 

        return x, objective


    def reverse(self, x, objective):
        # x : (bs,  C, H, W)
        canvas = torch.zeros_like(x)

        for i in range(x.size(-2)):
            for j in range(x.size(-1)):
                # h:  (bs, 2C, H, W)
                h = self.main(canvas)
                
                # s:  (bs,  C, H, W)
                shift, scale = h[:, ::2], F.sigmoid(h[:, 1::2] + 2.)
               
                tmp = x / scale
                tmp = tmp - shift

                canvas[:, :, i, j] = tmp[:, :, i, j]

                #objective += flatten_sum(torch.log(scale))
                
        objective -= torch.log(scale) 

        return canvas, objective

# ActNorm Layer with data-dependant init
class ActNorm(nn.Module):
    def __init__(self, num_features, logscale_factor=1., scale=1.):
        super(ActNorm, self).__init__()
        self.initialized = False
        self.logscale_factor = logscale_factor
        self.scale = scale
        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features, 1)))
        self.register_parameter('logs', nn.Parameter(torch.zeros(1, num_features, 1)))

    def forward(self, input, objective):
        input_shape = input.size()
        input = input.view(input_shape[0], input_shape[1], -1)

        if not self.initialized: 
            self.initialized = True
            unsqueeze = lambda x: x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = input.size(0) * input.size(-1)
            input_sum = input.sum(dim=0).sum(dim=-1)
            b = input_sum / sum_size * -1.
            vars = ((input - unsqueeze(b)) ** 2).sum(dim=0).sum(dim=1) / sum_size
            vars = unsqueeze(vars)
            logs = torch.log(self.scale / torch.sqrt(vars) + 1e-6) / self.logscale_factor
          
            self.b.data.copy_(unsqueeze(b).data)
            self.logs.data.copy_(logs.data)

        logs = self.logs * self.logscale_factor
        b = self.b
        
        output = (input + b) * torch.exp(logs)
        dlogdet = logs # torch.sum(logs) * input.size(-1) # c x h  

        return output.view(input_shape), objective + dlogdet

    def reverse(self, input, objective):
        assert self.initialized
        input_shape = input.size()
        input = input.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        output = input * torch.exp(-logs) - b
        dlogdet = logs # torch.sum(logs) * input.size(-1) # c x h  

        return output.view(input_shape), objective - dlogdet
    
# Flows 
# -----------------------------------------------------------------------------

class FlowNet(nn.Module):
    def __init__(self, num_layers=3):
        super(FlowNet, self).__init__()
        layers = []
        for i in range(num_layers):
            layers += [AF(1, 100)]
            if i != num_layers - 1 : 
                layers += [ActNorm(1)]

        #self.layers = [AF(1, 100) for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, objective):
        for layer in self.layers: 
            x, objective = layer(x, objective)

        return x, objective
        
    def reverse(self, x, objective):
        for layer in reversed(self.layers):
            x, objective = layer.reverse(x, objective)

        return x, objective


