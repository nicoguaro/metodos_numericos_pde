#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 10:41:08 2018

@author: nguarinz
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
Axes3D


def step_function(N, scale, X, Y, shape="crescent"):
    """Function that is 1 on a set and 0 outside of it"""
    shapes = ["crescent", "cylinder", "hexagon", "superquadric", "smiley"]

    if shape not in shapes:
        shape = "crescent"

    if shape == "cylinder":
        Z = np.ones_like(X)
        Z[X**2 + Y**2 < 0.5] = 0
        Z[X**2 + Y**2 > 2] = 0

    if shape == "superquadric":
        Z = np.ones_like(X)
        Z[np.abs(X)**0.5 + np.abs(Y)**0.5 > 1.5] = 0

    if shape == "hexagon":
        Z = np.ones_like(X)
        hexa = 2*np.abs(X) + np.abs(X - Y*np.sqrt(3)) +\
            np.abs(X + Y*np.sqrt(3))
        Z[hexa > 6] = 0

    if shape == "crescent":
        c = 2
        d = -1
        e = 1
        f = 0.5
        k = 1.2
        shift = 10
        Z = (c**2 - (X/e - d)**2 - (Y/f)**2)**2 + k*(c + d - X/e)**3 - shift
        Z = 1 - np.maximum(np.sign(Z), 0)

    if shape == "smiley":
        Z = np.ones_like(X)
        fac = 1.2
        x_eye = 0.5
        y_eye = 0.4
        bicorn = fac**2*(Y + 0.3)**2*(1 - fac**2*X**2) -\
                (fac**2*X**2 - 2*fac*(Y + 0.3) - 1)**2
        left_eye = (X + x_eye)**2/0.1 + (Y - y_eye)**2/0.4 - 1
        right_eye = (X - x_eye)**2/0.1 + (Y - y_eye)**2/0.4 - 1
        Z[X**2 + Y**2 > 2] = 0
        Z[bicorn > 0] = 0
        Z[left_eye < 0] = 0
        Z[right_eye < 0] = 0


    Z = scale * Z
    return Z


def heat_iter(num, ax, X, Y, Z, dt, ntime_anim, L, scale, plot_args):
    N, _ = X.shape
    dx = X[1, 0] - X[0, 0]
    # Solve the heat equation with zero boundary conditions
    for cont in range(ntime_anim):
        Z[1:N-1, 1:N-1] = Z[1:N-1, 1:N-1] + dt*(Z[2:N, 1:N-1] +
                             Z[0:N-2, 1:N-1] + Z[1:N-1, 0:N-2] +
                             Z[1:N-1, 2:N] - 4*Z[1:N-1, 1:N-1])/dx**2
    ax.cla()
    surf = ax.plot_surface(X, Y, Z, **plot_args)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-scale, scale)
    ax.view_init(elev=35, azim=120)
    return surf


def wave_iter(num, ax, X, Y, Z, Z0, dt, ntime_anim, L, scale, plot_args):
    N, _ = X.shape
    dx = X[1, 0] - X[0, 0]
    # Solve the wave equation with zero boundary conditions
    for cont in range(ntime_anim):
        Z_aux = Z.copy()
        Z[1:N-1, 1:N-1] = 2*Z[1:N-1, 1:N-1] - Z0[1:N-1, 1:N-1]  +\
                          (dt/dx)**2*(Z[2:N, 1:N-1] +
                           Z[0:N-2, 1:N-1] + Z[1:N-1, 0:N-2] +
                           Z[1:N-1, 2:N] - 4*Z[1:N-1, 1:N-1])
        Z0[:] = Z_aux[:]

    ax.cla()
    surf = ax.plot_surface(X, Y, Z, **plot_args)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-scale, scale)
    ax.view_init(elev=35, azim=120)
    return surf
