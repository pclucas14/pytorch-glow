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
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--depth', type=int, default=25) # 32
parser.add_argument('--n_levels', type=int, default=3) # 3 ?
parser.add_argument('--norm', type=str, default=None)
parser.add_argument('--permutation', type=str, default='shuffle')
parser.add_argument('--coupling', type=str, default='affine')
parser.add_argument('--data_dir', type=str, default='../pixelcnn-pp')
parser.add_argument('--n_bits_x', type=int, default=5)
parser.add_argument('--learntop', action='store_true')
args = parser.parse_args()
args.n_bins = 2 ** args.n_bits_x

tf = transforms.Compose([transforms.ToTensor(), 
                         lambda x: (x * 256.).byte(), 
                         lambda x: preprocess(x, args)])

train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
    download=True, transform=tf), batch_size=args.batch_size, shuffle=True)

test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                transform=tf), batch_size=args.batch_size, shuffle=True)

model = Glow((args.batch_size, 3, 32, 32), args)
model = model.cuda()
print(model)
print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
optim = optim.Adam(model.parameters(), lr=5e-4)

# data dependant init
if args.norm == 'actnorm': 
    init_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=tf), batch_size=512, shuffle=True, num_workers=1)
    
    model.eval()
    with torch.no_grad():
        for (img, _) in init_loader:
            img = img.cuda()
            _ = model(img, 0.)
            break

# training loop
for epoch in range(500):
    print('epoch %s' % epoch)
    model.train()
    avg_train_bits_x = 0.
    for i, (img, label) in enumerate(train_loader): 
        img = img.cuda()

        # log_det_jacobian cost (and some prior from Split OP)
        z, objective = model.forward_and_jacobian(img, 0.)
        
        # discretizing cost 
        objective += -np.log(args.n_bins) * np.prod(img.shape)

        # Generative loss
        nobj = - objective / img.shape[0]
        avg_train_bits_x += nobj / (np.log(2.) * np.prod(img.shape[1:]))

        optim.zero_grad()
        nobj.backward()
        optim.step()

    print('avg train bits per pixel {:.4f}'.format(avg_train_bits_x.item() / i))
    
    model.eval()
    avg_test_bits_x = 0.
    with torch.no_grad():
        for i, (img, label) in enumerate(test_loader): 
            img = img.cuda()

            # log_det_jacobian cost (and some prior from Split OP)
            z, objective = model.forward_and_jacobian(img, 0.)
            
            # discretizing cost 
            objective += -np.log(args.n_bins) * np.prod(img.shape)

            # Generative loss
            nobj = - objective / img.shape[0]
            avg_test_bits_x += nobj / (np.log(2.) * np.prod(img.shape[1:]))

        print('avg test bits per pixel {:.4f}'.format(avg_test_bits_x.item() / i))

    sample = model.sample()
    sample = sample.transpose(1, 3).cpu().data.numpy()[:10]
    sample = sample.reshape(10 * sample.shape[1], sample.shape[2], sample.shape[3])
    print('image  max %f, sample min %f'   % (img.max().item(), img.min().item()))
    print('sample max %f, sample min %f\n' % (sample.max().item(), sample.min().item()))
    sample = postprocess(sample, args)
    Image.fromarray(np.squeeze(sample)).save('samples/{}.png'.format(epoch))
    
