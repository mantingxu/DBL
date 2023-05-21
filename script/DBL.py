import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as Fun
import copy

output_dim = 24


class MyDBL(nn.Module):
    def __init__(self):
        super(MyDBL, self).__init__()
        self.linearOne = nn.Linear(36, output_dim)
        self.linearTwo = nn.Linear(36, output_dim)

    def forward(self, fx, fi):
        zx = self.linearOne(fx)
        zi = self.linearTwo(fi)
        tanh_layer = nn.Tanh()
        zx_prime = tanh_layer(zx)
        zi_prime = tanh_layer(zi)
        res = zx_prime * zi_prime
        # hadamard = zx * zi
        # res = tanh_layer(hadamard)
        return res


class MyMLP(nn.Module):
    def __init__(self, inputDim):
        super(MyMLP, self).__init__()
        self.linearOne = nn.Linear(inputDim, 32)
        self.linearTwo = nn.Linear(32, 16)
        self.linearThree = nn.Linear(16, 3)
        self.ReLU = nn.ReLU()

    def forward(self, f):
        o = self.linearOne(f)
        o = self.ReLU(o)
        o = self.linearTwo(o)
        o = self.ReLU(o)
        o = self.linearThree(o)
        return o


class DBLANet(nn.Module):
    def __init__(self, inputDim):
        super(DBLANet, self).__init__()
        self.dblOne = MyDBL()
        self.dblTwo = MyDBL()
        self.dblThree = MyDBL()
        self.mlp = MyMLP(inputDim)

    def forward(self, fx, f1, f2, f3):
        f1_prime = self.dblOne(fx, f1)
        f2_prime = self.dblTwo(fx, f2)
        f3_prime = self.dblThree(fx, f3)
        # print(f1_prime.shape)
        f1_prime = torch.squeeze(f1_prime, 0)
        f2_prime = torch.squeeze(f2_prime, 0)
        f3_prime = torch.squeeze(f3_prime, 0)
        final_f = torch.hstack((f1_prime, f2_prime, f3_prime))
        # print(final_f.shape)
        res = self.mlp(final_f)
        return res
