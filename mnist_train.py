import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils

import numpy as np
import pdb
import argparse
from PIL import Image

from invertible_layers import * 
from utils import * 


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--depth', type=int, default=10) # 32
parser.add_argument('--n_levels', type=int, default=3)
parser.add_argument('--norm', type=str, default=None)
parser.add_argument('--permutation', type=str, default='shuffle')
parser.add_argument('--coupling', type=str, default='affine')
parser.add_argument('--data_dir', type=str, default='../pixelcnn-pp')
parser.add_argument('--n_bins', type=int, default=2)
parser.add_argument('--n_bits_x', type=int, default=1)
parser.add_argument('--learntop', action='store_true')
args = parser.parse_args()

tf = transforms.Compose([transforms.Resize((32, 32)), 
                         transforms.ToTensor(), 
                         lambda x: (x * 256.).byte(), 
                         lambda x: preprocess(x, args)])

train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                    train=True, transform=tf), batch_size=args.batch_size, 
                        shuffle=True, num_workers=4)

test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                transform=tf), batch_size=args.batch_size, shuffle=True, num_workers=4)

model = Codec((args.batch_size, 1, 32, 32), args)
model = model.cuda()
print(model)
print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
optim = optim.Adam(model.parameters(), lr=5e-4)

for epoch in range(100):
    print('epoch %s' % epoch)
    for i, (img, label) in enumerate(train_loader): 
        img = img.cuda()
        
        # log_det_jacobian cost (and some prior from Split OP)
        z, objective = model.forward_and_jacobian(img, 0.)
        
        # discretizing cost 
        objective += -np.log(args.n_bins) * np.prod(img.shape)

        # Generative loss
        nobj = - objective / img.shape[0]
        bits_x = nobj / (np.log(2.) * np.prod(img.shape[1:]))

        optim.zero_grad()
        nobj.backward()
        optim.step()

        print('bits per pixel {:.4f}'.format(bits_x.item()))

    sample = model.sample().cpu().data.numpy()[0,0]
    sample = postprocess(sample, args)
    Image.fromarray(np.squeeze(sample)).save('samples/{}.png'.format(epoch))
    
