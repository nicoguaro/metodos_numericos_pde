#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve the "heat" equation using Jacobi iteration and finite differences 

@author: Nicolas Guarin-Zapata
"""
import numpy as np
import matplotlib.pyplot as plt

def jacobi_iter(A):
    """Compute a Jacobi iteration for the matrix A using a central
    finite difference
    """
    B = A.copy()
    B[1:-1, 1:-1] = 0.25*(B[0:-2, 1:-1] + B[2:, 1:-1] + B[1:-1, 0:-2] +
                            B[1:-1, 2:])
    return B


def compute_niter(A, niter):
    for n in range(niter):
        A = jacobi_iter(A)
    return A
    
nx = 50
ny = 50
x_vec = np.linspace(-0.5, 0.5, nx)
y_vec = np.linspace(-0.5, 0.5, ny)
A = np.zeros((nx, ny))
A[:, 0] = 1 - x_vec
A[0, :] = 1 - y_vec

nvec = [100, 1000, 10000, 100000]
for num, niter in enumerate(nvec):
    B = compute_niter(A, niter)
    plt.subplot(2,2, num+1)
    plt.contourf(B, cmap='hot')
    plt.title(r'$N=%d$'%niter)
    plt.axis('image')

plt.tight_layout()
plt.show()
