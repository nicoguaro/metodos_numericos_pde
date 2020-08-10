# -*- coding: utf-8 -*-
"""
Solucion de la ecuacion de onda en 1D usando diferencias finitas


@author: Nicolas Guarin-Zapata
@date: Marzo, 2019
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#%% Condiciones iniciales
def desp_inicial(x):
    """Desplazamiento inicial de la cuerda"""
    return np.exp(-1000*(x - 0.3)**2)


def vel_inicial(x):
    """Velocidad inicial de la cuerda"""
    return np.zeros_like(x)

#%% Carga
def carga(x, t):
    return np.zeros_like(x)


#%% Parametros de entrada para 
densidad = 7800 # kg/m^3
diametro = 0.66e-3  # m
area = (np.pi/4) * diametro**2 # m^2
tension = 81.732 # N
c = np.sqrt(tension/(densidad * area))
npuntos = 200
longitud = 0.6
x = np.linspace(0, longitud, npuntos)
dx = x[1] - x[0]
dt = 1.0 * dx/c
alpha = c * dt/dx
niteraciones = 300

#%% Solucion
solucion = np.zeros((niteraciones, npuntos))

# Condiciones iniciales
solucion[0, :] = desp_inicial(x)
#solucion[1, :] = desp_inicial(x) - dt * vel_inicial(x)
solucion[1, 1:-1] = 0.5*alpha**2 * (solucion[0, 2:] + solucion[0, 0:-2] 
                  - 2*solucion[0, 1:-1]) + solucion[0, 1:-1] \
                  - dt*vel_inicial(x)[:1:-1] + dt**2 * carga(x, 0)[1:-1]

# Solucion para cada tiempo mayor a 2 * dt
for cont_t in range(2, niteraciones):
    q = carga(x, cont_t * dt)
    solucion[cont_t, 1:-1] = alpha**2 * (solucion[cont_t - 1, 2:] +
                solucion[cont_t - 1, 0:-2] - 2*solucion[cont_t - 1, 1:-1])\
              + 2*solucion[cont_t - 1, 1:-1] - solucion[cont_t - 2, 1:-1]\
              + dt**2 * q[1:-1]
    


#%% Animacion
max_val = max(np.max(solucion), -np.min(solucion))
fig, ax = plt.subplots()
line, = ax.plot(x, solucion[0, :])
def update(data):
    line.set_ydata(data)
    return line,

ani = animation.FuncAnimation(fig, update, solucion, interval=niteraciones,
                              repeat=True)
plt.grid()
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Cuerda vibrando')
ax.set_ylim(-1.2*max_val, 1.2*max_val)
plt.show()

