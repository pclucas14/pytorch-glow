# inspired by https://github.com/ikostrikov/pytorch-flows/blob/master/flow_test.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn

import numpy as np
import unittest
import pdb

from invertible_layers import * 

EPS = 1e-5
BATCH_SIZE = 17
NUM_CHANNELS = 64
H = 32
W = 48


class TestFlow(unittest.TestCase):
    def test_shuffle(self):
        x = torch.randn(BATCH_SIZE, NUM_CHANNELS, H, W)
        layer = Shuffle(NUM_CHANNELS)
        log_det = torch.randn(BATCH_SIZE)

        y, inv_log_det = layer.forward_(x.clone(), log_det.clone())
        x_, log_det_   = layer.reverse_(y.clone(), inv_log_det.clone())

        self.assertTrue((log_det_ - log_det).abs().max() < EPS, 
                        'Shuffle Layer det is not zero.')

        self.assertTrue((x - x_).abs().max() < EPS, 'Shuffle Layer is wrong')

    def test_reverse(self):
        x = torch.randn(BATCH_SIZE, NUM_CHANNELS, H, W)
        layer = Reverse(NUM_CHANNELS)
        log_det = torch.randn(BATCH_SIZE)

        y, inv_log_det = layer.forward_(x.clone(), log_det.clone())
        x_, log_det_   = layer.reverse_(y.clone(), inv_log_det.clone())

        self.assertTrue((log_det_ - log_det).abs().max() < EPS, 
                        'Shuffle Layer det is not zero.')

        self.assertTrue((x - x_).abs().max() < EPS, 'Shuffle Layer is wrong')

    def test_conv(self):
        x = torch.randn(BATCH_SIZE, NUM_CHANNELS, H, W)
        layer = Invertible1x1Conv(NUM_CHANNELS)
        log_det = torch.randn(BATCH_SIZE)

        y, inv_log_det = layer.forward_(x.clone(), log_det.clone())
        x_, log_det_   = layer.reverse_(y.clone(), inv_log_det.clone())

        self.assertTrue((log_det_ - log_det).abs().max() < EPS, 
                        'Conv Layer det is not zero.')

        self.assertTrue((x - x_).abs().max() < EPS, 'Conv Layer is wrong')
        self.assertTrue((log_det - inv_log_det).abs().max() > 0.01 * EPS, 'Determinant was not changed!')        
    def test_squeeze(self):
        x = torch.randn(BATCH_SIZE, NUM_CHANNELS, H, W)
        layer = Squeeze([int(y) for y in x.size()])
        log_det = torch.randn(BATCH_SIZE)

        y, inv_log_det = layer.forward_(x.clone(), log_det.clone())
        x_, log_det_   = layer.reverse_(y.clone(), inv_log_det.clone())

        self.assertTrue((log_det_ - log_det).abs().max() < EPS, 
                        'Squeeze Layer det is not zero.')

        self.assertTrue((x - x_).abs().max() < EPS, 'Squeeze Layer is wrong')

    def test_split(self):
        x = torch.randn(BATCH_SIZE, NUM_CHANNELS, H, W)
        layer = Split([int(y) for y in x.size()])
        log_det = torch.randn(BATCH_SIZE)

        y, inv_log_det = layer.forward_(x.clone(), log_det.clone())
        x_, log_det_   = layer.reverse_(y.clone(), inv_log_det.clone(), use_stored_sample=True)

        self.assertTrue((log_det_ - log_det).abs().max() < 1e-2, 
                        'Squeeze Layer det is not zero.')

        self.assertTrue((x - x_).abs().max() < EPS, 'Squeeze Layer is wrong')
        self.assertTrue((log_det - inv_log_det).abs().max() > EPS, 'Determinant was not changed!')        
    def test_add(self):
        x = torch.randn(BATCH_SIZE, NUM_CHANNELS, H, W)
        layer = AdditiveCoupling(NUM_CHANNELS)
        log_det = torch.randn(BATCH_SIZE)

        y, inv_log_det = layer.forward_(x.clone(), log_det.clone())
        x_, log_det_   = layer.reverse_(y.clone(), inv_log_det.clone())

        self.assertTrue((log_det_ - log_det).abs().max() < EPS, 
                        'Additive Coupling Layer det is not zero.')

        self.assertTrue((x - x_).abs().max() < EPS, 'Additive Coupling Layer is wrong')

    def test_affine(self):
        x = torch.randn(BATCH_SIZE, NUM_CHANNELS, H, W)
        layer = AffineCoupling(NUM_CHANNELS)
        log_det = torch.randn(BATCH_SIZE)

        # import pdb; pdb.set_trace()
        y, inv_log_det = layer.forward_(x.clone(), log_det.clone())
        x_, log_det_   = layer.reverse_(y.clone(), inv_log_det.clone())

        self.assertTrue((log_det_ - log_det).abs().max() < 1e-3, 
                        'affine coupling layer det is not zero.')

        self.assertTrue((x - x_).abs().max() < EPS, 'affine coupling layer is wrong')
        self.assertTrue((log_det - inv_log_det).abs().max() > EPS, 'determinant was not changed!')        
        
    def test_actnorm(self):
        x = torch.randn(BATCH_SIZE, NUM_CHANNELS, H, W)
        layer = ActNorm(NUM_CHANNELS)
        log_det = torch.randn(BATCH_SIZE)

        # import pdb; pdb.set_trace()
        y, inv_log_det = layer.forward_(x.clone(), log_det.clone())
        x_, log_det_   = layer.reverse_(y.clone(), inv_log_det.clone())

        self.assertTrue((log_det_ - log_det).abs().max() < EPS, 
                        'actnorm layer det is not zero.')

        self.assertTrue((x - x_).abs().max() < EPS, 'actnorm layer is wrong')
        self.assertTrue((log_det - inv_log_det).abs().max() > EPS, 'determinant was not changed!')        



if __name__ == '__main__':
    unittest.main()
