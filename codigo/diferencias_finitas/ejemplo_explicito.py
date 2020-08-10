# -*- coding: utf-8 -*-
"""
Solucion de la ecuacion de calor usando un esquema explicito

@author: Nicolas Guarin-Zapata
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from diferencias import resolver_explicito


niter = 10000
nx = 50
alpha = 1.0
fuente = lambda x: 1

x = np.linspace(-1, 1, nx)
np.random.seed(201)
u_ini = 0.5 * (1 + x) + 0.1 * np.random.normal(size=nx)
u_a = 0.0
u_b = 1.0
u_ini[0] = u_a
u_ini[-1] = u_b
k_iter = 0.4
dx = x[1] - x[0]
dt = k_iter * dx**2/alpha
t = np.linspace(0, niter*dt, niter)
U = resolver_explicito(niter, u_ini, alpha, dt, x, fuente)


#%% Animacion
max_val = max(np.max(U), -np.min(U))
fig, ax = plt.subplots()
line, = ax.plot(x, U[0, :])
def update(data):
    line.set_ydata(data)
    return line,

ani = animation.FuncAnimation(fig, update, U, interval=niter//100,
                              repeat=True)
plt.grid()
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Temperatura en una varilla')
plt.show()