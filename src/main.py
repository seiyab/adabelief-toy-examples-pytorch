import numpy as np
import torch

from .utils import compose, kw
from .problems import toy_problems, fig3d_problem
from .optimizers import fig3_optims, fig3d_optims
from .trajectory import trajectory
from .plot import plot_trajectory

def main():
    for (problem_name, f, init, domain) in toy_problems:
        adabelief_trajectory = trajectory(f, init, fig3_optims['AdaBelief'], 3*10**4, 10**-3)
        ts = {
            opt_name: trajectory(f, init, opt, adabelief_trajectory.shape[0])
            for opt_name, opt in fig3_optims.items()
            if opt_name != 'AdaBelief'
        }
        ts['AdaBelief'] = adabelief_trajectory

        fig = plot_trajectory(f, domain, ts)
        fig.save(f"out/{problem_name}.gif", fps=30)

    problem_name, f, init, domain = fig3d_problem
    adabelief_trajectory = trajectory(f, init, fig3d_optims['AdaBelief'], 3*10**4, 10**-3)
    ts = {
        opt_name: trajectory(f, init, opt, adabelief_trajectory.shape[0])
        for opt_name, opt in fig3d_optims.items()
        if opt_name != 'AdaBelief'
    }
    ts['AdaBelief'] = adabelief_trajectory

    fig = plot_trajectory(f, domain, ts)
    fig.save(f"out/{problem_name}.gif", fps=30)
    


