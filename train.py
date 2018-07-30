import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils

import numpy as np
import pdb
import argparse
import time

from invertible_layers import * 
from utils import * 

# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# training
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--depth', type=int, default=10) 
parser.add_argument('--n_levels', type=int, default=3) 
parser.add_argument('--norm', type=str, default='actnorm')
parser.add_argument('--permutation', type=str, default='shuffle')
parser.add_argument('--coupling', type=str, default='affine')
parser.add_argument('--n_bits_x', type=int, default=8)
parser.add_argument('--n_epochs', type=int, default=2000)
parser.add_argument('--learntop', action='store_true')
parser.add_argument('--n_warmup', type=int, default=20, help='number of warmup epochs')
parser.add_argument('--lr', type=float, default=1e-3)
# logging
parser.add_argument('--print_every', type=int, default=500, help='print NLL every _ minibatches')
parser.add_argument('--test_every', type=int, default=5, help='test on valid every _ epochs')
parser.add_argument('--save_every', type=int, default=5, help='save model every _ epochs')
parser.add_argument('--data_dir', type=str, default='../pixelcnn-pp')
parser.add_argument('--save_dir', type=str, default='exps', help='directory for log / saving')
parser.add_argument('--load_dir', type=str, default=None, help='directory from which to load existing model')
args = parser.parse_args()
args.n_bins = 2 ** args.n_bits_x

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# loading / dataset preprocessing
tf = transforms.Compose([transforms.ToTensor(), 
                         lambda x: x + torch.zeros_like(x).uniform_(0., 1./args.n_bins)])

train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
    download=True, transform=tf), batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)

test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
    transform=tf), batch_size=args.batch_size, shuffle=False, num_workers=10)

# construct model and ship to GPU
model = Glow_((args.batch_size, 3, 32, 32), args).cuda()
print(model)
print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))

# set up the optimizer
optim = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=45, gamma=0.1)

# data dependant init
init_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
    download=True, transform=tf), batch_size=512, shuffle=True, num_workers=1)

with torch.no_grad():
    model.eval()
    for (img, _) in init_loader:
        img = img.cuda()
        objective = torch.zeros_like(img[:, 0, 0, 0])
        _ = model(img, objective)
        break

# once init is done, we leverage Data Parallel
model = nn.DataParallel(model).cuda()
start_epoch = 0

# load trained model if necessary (must be done after DataParallel)
if args.load_dir is not None: 
    model, optim, start_epoch = load_session(model, optim, args)

# training loop
# ------------------------------------------------------------------------------
for epoch in range(start_epoch, args.n_epochs):
    print('epoch %s' % epoch)
    model.train()
    avg_train_bits_x = 0.
    num_batches = len(train_loader)
    for i, (img, label) in enumerate(train_loader):
        t = time.time()
        img = img.cuda() 
        objective = torch.zeros_like(img[:, 0, 0, 0])
       
        # discretizing cost 
        objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
        
        # log_det_jacobian cost (and some prior from Split OP)
        z, objective = model(img, objective)

        nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
        
        # Generative loss
        nobj = torch.mean(nll)

        optim.zero_grad()
        nobj.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        optim.step()
        avg_train_bits_x += nobj.item()

        # update learning rate
        new_lr = float(args.lr * min(1., (i + epoch * num_batches) / (args.n_warmup * num_batches)))
        for pg in optim.param_groups: pg['lr'] = new_lr

        if (i + 1) % args.print_every == 0: 
            print('avg train bits per pixel {:.4f}'.format(avg_train_bits_x / args.print_every))
            avg_train_bits_x = 0.
            sample = model.module.sample()
            grid = utils.make_grid(sample)
            utils.save_image(grid, '../glow/samples/cifar_Test_{}_{}.png'.format(epoch, i // args.print_every))

        print('iteration took {:.4f}'.format(time.time() - t))
        
    # test loop
    # --------------------------------------------------------------------------
    if (epoch + 1) % args.test_every == 0:
        model.eval()
        avg_test_bits_x = 0.
        with torch.no_grad():
            for i, (img, label) in enumerate(test_loader): 
                img = img.cuda() 
                objective = torch.zeros_like(img[:, 0, 0, 0])
               
                # discretizing cost 
                objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
                
                # log_det_jacobian cost (and some prior from Split OP)
                z, objective = model(img, objective)

                nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
                
                # Generative loss
                nobj = torch.mean(nll)
                avg_test_bits_x += nobj

            print('avg test bits per pixel {:.4f}'.format(avg_test_bits_x.item() / i))

        sample = model.module.sample()
        grid = utils.make_grid(sample)
        utils.save_image(grid, '../glow/samples/cifar_Test_{}.png'.format(epoch))
    
    if (epoch + 1) % args.save_every == 0: 
        save_session(model, optim, args, epoch)

        
