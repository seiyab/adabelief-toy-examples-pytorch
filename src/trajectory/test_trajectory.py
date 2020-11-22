import unittest

import numpy as np
import torch
from torch import optim

from ..utils import compose, kw
from . import trajectory

class TestTrajectory(unittest.TestCase):
    def test_trajectory(self):
        f = compose(torch.sum, torch.abs)
        init = np.array([-1., -1.])
        opt = kw(optim.SGD, lr=1e-3)

        t = trajectory(f, init, opt, 10)
        tt = torch.tensor(t)

        self.assertTrue(all(t[0] == init))
        self.assertLess(f(tt[-1,:]).tolist(), f(tt[0,:]).tolist())

if __name__ == "__main__":
    unittest.main()

