import numpy as np

import torch

def trajectory(f, init, newopt, step):
    params = torch.tensor(init, requires_grad=True, dtype=torch.float64)
    opt = newopt([params])
    footprints = [np.copy(params.data.numpy())]
    for _ in range(step):
        opt.zero_grad()
        z = f(params)
        z.backward()
        opt.step()
        footprints.append(np.copy(params.data.numpy()))
    return np.array(footprints)

