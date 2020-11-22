import numpy as np
import torch
from torch import optim
import adabelief_pytorch

from .utils import compose, kw
from .problem import Problem
from .trajectory import trajectory

def main():
    f=compose(torch.sum, torch.abs)
    init=np.array([-1., 1.])
    opt = kw(optim.SGD, lr=1e-3, momentum=0.9)

    t = trajectory(f, init, opt, 10)
    print(t)

