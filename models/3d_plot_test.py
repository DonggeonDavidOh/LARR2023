import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib import animation

fig = plt.figure(facecolor='black')
ax = plt.axes(projection = "3d")

u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)
r = 4

ax.set_xlim(0, 60)
ax.set_ylim(0, 60)
ax.set_zlim(0, 60)

x0 = r * np.outer(np.cos(u), np.sin(v)) + 10
y0 = r * np.outer(np.sin(u), np.sin(v)) + 10
z0 = r * np.outer(np.ones(np.size(u)), np.cos(v)) + 50

surface_color = "tab:blue"

def init():
    ax.plot_surface(x0, y0, z0, color=surface_color)
    return fig,

def animate(i):
    # remove previous collections
    ax.collections.clear()
    # add the new sphere
    ax.plot_surface(x0 + i, y0 + i, z0 + i, color=surface_color)
    return fig,

ani = animation. FuncAnimation(fig, animate, init_func = init, frames = 90, interval = 300)
ani.save("3d_test" + ".mp4", dpi=300, writer=animation.writers["ffmpeg"](fps=10))

plt.show()