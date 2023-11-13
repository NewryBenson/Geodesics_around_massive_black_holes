import os

import numpy as np
from numpy import pi, sqrt, sin, cos, floor
from time import time
import PIL.Image as Image
from multiprocessing import Pool
from scipy.integrate import solve_ivp as solve
from warnings import filterwarnings
import matplotlib.pyplot as plt
import pylab as pl
plt.rcParams['text.usetex'] = True
filterwarnings("ignore")

def _dSdl(l, s, m):
    t, r, theta, phi, pt, pr, ptheta, pphi = s
    return [-1 / (1 - (2 * m / r)) * pt,
            (1 - (2 * m / r)) * pr,
            1 / (r ** 2) * ptheta,
            1 / ((r ** 2) * sin(theta) ** 2) * pphi,
            0,
            -(1 / 2) * (1 / ((1 - (2 * m / r)) ** 2) * (2 * m / (r ** 2)) * pt ** 2 + (
                    2 * m / (r ** 2)) * pr ** 2 - 2 / (r ** 3) * ptheta ** 2 - 2 / (
                                (r ** 3) * sin(theta) ** 2) * pphi ** 2),
            cos(theta) / ((sin(theta) ** 3) * (r ** 2)) * pphi ** 2,
            0]


def _solver(t0, r0, theta0, phi0, pt0, pr0, ptheta0, pphi0, l_span, M):
    s_0 = (t0, r0, theta0, phi0, pt0, pr0, ptheta0, pphi0)
    sol = solve(_dSdl, l_span, s_0, method='DOP853', args=[M], max_step=0.1)
    return sol


def get_plot_data(r,theta,phi, localx, localy, localz, mass):
    """Returns data for plotting the ray path."""
    pr0, ptheta0, pphi0 = -localx / (
            1 - 2 * mass / r), -localz * r / sqrt(
        1 - 2 * mass / r), localy * r * sin(theta) / sqrt(
        1 - 2 * mass / r)
    sol = _solver(0, r, theta, phi, -1, pr0, ptheta0, pphi0, (0, 200),
                       mass)
    return sol

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
'''
number_of_rays = 100
for i in range(number_of_rays):
    sol = get_plot_data(10, pi/2, 0, 1, i/number_of_rays, 0, 1)["y"]
    ax.plot(sol[3], sol[1])
    ax.set_ylim([0, 10])
ax.set_rticks([int(2), int(4), 6, 8,10])
ax.set_yticklabels([str(2), str(4), '6' , "8" , 'black hole masses'])
plt.title("Rays around a black hole")
plt.show()
'''
y = 0.4647580015

sol = get_plot_data(10, pi/2, 0, 1, 1/5, 0, 1)["y"]
ax.plot(sol[3], sol[1], color = "r", label="plunging orbit")
sol = get_plot_data(10, pi/2, 0, 1,  y, 0, 1)["y"]
ax.plot(sol[3], sol[1], color = "g", label="circular orbit")
sol = get_plot_data(10, pi/2, 0, 1, 1, 0, 1)["y"]
ax.plot(sol[3], sol[1], color = "b", label="scattering orbit")
sol = get_plot_data(10, pi/2, 0, 1, 1/5, 0, 0)["y"]
ax.plot(sol[3], sol[1], '--', color = "r", alpha = 0.5 )
sol = get_plot_data(10, pi/2, 0, 1,  y, 0, 0)["y"]
ax.plot(sol[3], sol[1], '--',color = "g", alpha = 0.5)
sol = get_plot_data(10, pi/2, 0, 1, 1, 0, 0)["y"]
ax.plot(sol[3], sol[1], '--',color = "b", alpha = 0.5)
ax.set_ylim([0, 10])
ax.set_rticks([int(2), int(3), np.sqrt(27), int(10)])
ax.set_yticklabels([str(2), str(3), r'$\sqrt{27}$', "black hole masses"])
ax.text(100,100, "Black hole masses")
plt.legend()
plt.title("Rays around a black hole")
plt.show()