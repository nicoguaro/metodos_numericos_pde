"""
Illustration of the heat equation

Solve the wave equation using finite differences and Forward Euler.

Based on: https://commons.wikimedia.org/wiki/File:Heat_eqn.gif
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from FD_iter import step_function, wave_iter
Axes3D


N = 50  # Grid points
L = 2.5  # Box size
end_time = 5.0
nframes = 50
scale = 2
CFL = 0.25
X, Y = np.mgrid[-L:L:N*1j, -L:L:N*1j]
dx = X[1, 0] - X[0, 0]
dt = CFL*dx
time = np.arange(0, end_time, dt)
ntime = time.shape[0]
ntime_anim = int(ntime/nframes)
Z0 = step_function(N, scale, X, Y, shape="crescent")
Z0 = gaussian_filter(Z0, sigma=4)
Z = np.zeros_like(Z0)
# First iteration
Z[1:N-1, 1:N-1] = Z0[1:N-1, 1:N-1] + 0.5*(dt/dx)**2*(Z0[2:N, 1:N-1] +
                       Z0[0:N-2, 1:N-1] + Z0[1:N-1, 0:N-2] +
                       Z0[1:N-1, 2:N] - 4*Z0[1:N-1, 1:N-1])

#%% Animation
plot_args = {'rstride': 1, 'cstride': 1, 'cmap':"summer_r",
             'linewidth': 0.1, 'antialiased': True, 'edgecolor': '#1e1e1e',
             'shade': True, 'alpha': 1.0, 'vmin': 0, 'vmax':scale}
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ani = animation.FuncAnimation(fig, wave_iter, range(nframes), blit=False,
                              fargs=(ax, X, Y, Z, Z0, dt, ntime_anim, L, scale,
                                     plot_args))
plt.show()

