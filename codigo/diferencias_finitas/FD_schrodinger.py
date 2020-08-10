#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resuelve la ecuacion de Schrodinger independiente del tiempo

@author: Nicolas Guarin-Zapata
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

L = 8
N = 1001
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

T = -0.5*diags([-2., 1., 1.], [0,-1, 1], shape=(N, N))/dx**2
U_vec = 0.5*x**2
U = diags([U_vec], [0])

H = T + U

vals, vecs = eigsh(H, which='SA')

print(np.round(vals, 6))
print([k + 0.5 for k in range(6)])

for k in range(4):
    vec = vecs[:, k]
    mag = np.sqrt(np.dot(vecs[:, k],vecs[:, k]))
    vec = vec/mag
    plt.plot(x, vec, label=r"$n=%i$"%k)

plt.xlabel(r"$x$")
plt.ylabel(r"$\psi(x)$")
plt.legend()    
plt.show()
