"""
Illustration of the heat equation

Solve the heat equation using finite differences and Forward Euler.

Based on: https://commons.wikimedia.org/wiki/File:Heat_eqn.gif
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from FD_iter import step_function, heat_iter
Axes3D



N = 50  # Grid points
L = 2.5  # Box size
X, Y = np.mgrid[-L:L:N*1j, -L:L:N*1j]
scale = 2
Z = step_function(N, scale, X, Y, shape="crescent")
CFL = 0.125
dx = X[1, 0] - X[0, 0]
dy = dx
dt = CFL*dx**2
end_time = 0.1
time = np.arange(0, end_time, dt)
nframes = 50
ntime = time.shape[0]
ntime_anim = int(ntime/nframes)

#%% Animation
plot_args = {'rstride': 1, 'cstride': 1, 'cmap':"autumn_r",
             'linewidth': 0.1, 'antialiased': True, 'edgecolor': '#1e1e1e',
             'shade': True, 'alpha': 1.0, 'vmin': 0, 'vmax':scale}
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ani = animation.FuncAnimation(fig, heat_iter, range(nframes), blit=False,
                              fargs=(ax, X, Y, Z, dt, ntime_anim, L, scale,
                                     plot_args))
plt.show()

