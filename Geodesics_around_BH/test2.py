#imports
import numpy as np
import time
import math
import PIL.Image as Image
import tqdm.std as tqdm
import multiprocess.pool as Pool
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as Solve


def dSdl(l, S, M):
    t, r, theta, phi, pt, pr, ptheta, pphi = S
    return [-1/(1-(2*M/r))*pt,
            (1-(2*M/r))*pr,
            1/r**2*ptheta,
            1/(r**2*np.sin(theta)**2)*pphi,
            0,
            -(1/2)*(1/(1-(2*M/r))**2 * (2*M/(r**2))*pt**2 + (2*M/(r**2))*pr**2 -2/(r**3)*ptheta**2 -2/(r**3*np.sin(theta)**2) * pphi**2),
            np.cos(theta)*(np.sin(theta))**(-3) * r**(-2) * pphi**2,
            0]

#l_span should be an (l0, lfinal) object
def solver(t0, r0, theta0, phi0, pt0, pr0, ptheta0, pphi0, l_span, M):
    S_0 = (t0, r0, theta0, phi0, pt0, pr0, ptheta0, pphi0)
    sol = Solve(dSdl, l_span, S_0, method='RK45', args=[M])
    return sol



sol = solver(0, 3, np.pi/2, 0 , 1, 0, 100, np.pi/2, (0, 200), 1)

print(sol)
