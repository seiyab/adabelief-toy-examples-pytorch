import numpy as np

import torch

def trajectory(f, init, newopt, max_step, stop_at=None):
    params = torch.tensor(init, requires_grad=True, dtype=torch.float64)
    opt = newopt([params])
    footprints = [np.copy(params.detach().numpy())]
    for i in range(max_step):
        opt.zero_grad()
        z = f(params)
        z.backward()
        opt.step()
        footprints.append(np.copy(params.detach().numpy()))
        if stop_at is not None and z.detach().numpy().tolist() < stop_at:
            break
    return np.array(footprints)

