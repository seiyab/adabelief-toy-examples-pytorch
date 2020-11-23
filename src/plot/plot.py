import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.ticker as ticker
import numpy as np
import torch

def plot_trajectory(f, domain, trajectories):
    fig, ax = plt.subplots()
    ax.set_xlim(*domain)
    ax.set_ylim(*domain)
    ax.set_aspect(1)

    grid = 300
    l = np.linspace(*domain, num=grid)
    x, y = np.meshgrid(l, l)
    xy = np.stack((x, y), axis=-1)
    z = f(torch.tensor(xy)).numpy()
    lct = ticker.LinearLocator() if z.max() < 100 else ticker.LogLocator()
    ax.contour(x, y, z, locator=lct)

    frames = 200
    steps = min(t.shape[0] for t in trajectories.values())
    step_per_frame = steps / frames

    lines = {
            name: (ax.plot([], [], label=name))[0]
            for name in trajectories.keys()
        }
    dots = {
            name: (ax.plot([], [], marker='.', color=lines[name].get_color()))[0]
            for name in trajectories.keys()
        }
    ax.legend(loc='upper left')

    def animate(frame):
        for name, trajectory in trajectories.items():
            step = min(int(frame * step_per_frame), steps)
            lines[name].set_data(trajectory[:step,0], trajectory[:step,1])
            dots[name].set_data([trajectory[step,0]], [trajectory[step,1]])
        return lines.values()

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=20, blit=True)

    return anim

