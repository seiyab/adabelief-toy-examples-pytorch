from torch import optim
from adabelief_pytorch import AdaBelief

from ..utils import kw


fig3_optims = {
    "SGD + Momentum": kw(optim.SGD, lr=10**-3, momentum=0.9, dampening=0.9),
    "SGD + Momentum (α=10^-6)": kw(optim.SGD, lr=10**-6, momentum=0.9),
    "Adam": optim.Adam,
    "AdaBelief": kw(AdaBelief, rectify=False),
}

