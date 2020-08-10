# -*- coding: utf-8 -*-
"""
Funciones para la solución de ecuaciones diferenciales usando
diferencias finitas.

@author: Nicolas Guarín-Zapata
"""
import numpy as np
from scipy.sparse import diags, linalg


def mat_dif(n, F):
    """
    Genera la matriz de difusión para un problema
    con n puntos con una separación h.
    """
    mat = diags([-F, 1 + 2*F, -F], [-1, 0, 1], shape=(n, n), format="csr")
    mat[0, 0] = 1
    mat[n - 1, n - 1] = 1
    mat[0, 1] = mat[n - 1, n - 2] = 0
    return mat


def actualizar_sol(u, alpha, dt, x, fuente):
    """
    Actualiza la iteración para diferencias finitas
    explícitas para la ecuación de calor.
    """
    dx = x[1] - x[0]
    F = alpha * dt/dx**2
    u_aux = u.copy()
    u_aux[1:-1] = u_aux[1:-1] + F * (u_aux[2:] - 2*u[1:-1] + u[0:-2])  + \
                  dt * fuente(x[1:-1])
    return u_aux


def resolver_explicito(niter, u_ini, alpha, dt, x, fuente):
    """
    Resuelve la ecuación de calor usando un esquema
    explícito de diferencias finitas.
    """
    nx = len(u_ini)
    U = np.zeros((niter, nx))
    U[0, :] = u_ini
    for cont in range(1, niter):
        U[cont, :] = actualizar_sol(U[cont - 1, :], alpha, dt, x, fuente)
    return U


def resolver_implicito(niter, u_ini, alpha, dt, x, fuente):
    """
    Resuelve la ecuación de calor usando un esquema
    explícito de diferencias finitas.
    """
    nx = len(u_ini)
    dx = x[1] - x[0]
    F = alpha * dt/dx**2
    A_mat = mat_dif(nx, F)
    U = np.zeros((niter, nx))
    U[0, :] = u_ini
    for cont in range(1, niter):
        b = U[cont - 1] + dt*fuente(x)
        b[0] = u_ini[0]
        b[-1] = u_ini[-1]
        U[cont, :] = linalg.spsolve(A_mat, b)       
    return U