import torch

from ..utils import compose, kw

def from2d(f):
    return lambda t: f(t[...,0], t[...,1])

fa = compose(kw(torch.sum, dim=-1), torch.abs)
fb = from2d(lambda x, y: torch.abs(x + y) + torch.abs(x - y) / 10)
fc = from2d(lambda x, y: (x + y)**2 + (x-y)**2 / 10)
fd = from2d(lambda x, y: torch.abs(x)/10 + torch.abs(y))
beale = from2d(lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2)
rosenbrock = from2d(lambda x, y: (1 - x)**2 + 100*(y - x**2)**2)

toy_problems = [
        # (name, function, initial_position, domain)
        (
            'fig.3(a)',
            fa,
            [-2.5, 0],
            (-3, 3),
        ),
        (
            'fig.3(b)',
            fb,
            [2.5, 0],
            (-3, 3),
        ),
        (
            'fig.3(c)',
            fc,
            [2.5, 0],
            (-3, 3),
        ),
        (
            'fig.3(e)',
            beale,
            [-2.5, -2.5],
            (-4, 4),
        ),
        (
            'fig.3(g)',
            rosenbrock,
            [-4, -4],
            (-6, 6),
        ),
    ]

fig3d_problem = ('fig.3(d)', fd, [0.5, -1.5], (-3, 3))
