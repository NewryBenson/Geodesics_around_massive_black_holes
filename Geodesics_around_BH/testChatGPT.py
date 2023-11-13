import os

import numpy as np
from numpy import pi, sqrt, sin, cos
from time import time
import PIL.Image as Image
from multiprocessing import Pool
from scipy.integrate import solve_ivp as solve
from warnings import filterwarnings
import matplotlib.pyplot as plt

filterwarnings("ignore")


class Background:
    def __init__(self, image: Image):
        self._image = image

    @property
    def image(self):
        """Returns the image object of the background."""
        return self._image

    def get_angle_value(self, theta, phi):
        """Get angle value from the background image."""
        return self.image.load()[self.image.size[0]-np.floor(phi / (2 * pi) * self.image.size[0])%self.image.size[0]-1, np.ceil(theta / pi * self.image.size[1])%self.image.size[1]]


class BlackHole:
    def __init__(self, mass: float):
        self._mass = mass

    @property
    def mass(self):
        """Returns the mass of the black hole."""
        return self._mass


class Camera:
    def __init__(self, r: float, theta: float, phi: float, distance: float, resolution: int, size: float):
        self._r = r
        self._theta = theta
        self._phi = phi
        self._distance = distance
        self._resolution = resolution
        self._size = size

    @property
    def r(self):
        """Returns the r coordinate."""
        return self._r

    @property
    def theta(self):
        """Returns the theta coordinate."""
        return self._theta

    @property
    def phi(self):
        """Returns the phi coordinate."""
        return self._phi

    @property
    def distance(self):
        """Returns the distance to the screen."""
        return self._distance

    @property
    def resolution(self):
        """Returns the resolution of the screen."""
        return self._resolution

    @property
    def size(self):
        """Returns the size of the screen."""
        return self._size


class Ray:
    def __init__(self, pixelx: int, pixely: int, camera: Camera, background: Background, blackhole: BlackHole):
        self._pixelx = pixelx
        self._pixely = pixely
        self._camera = camera
        self._background = background
        self._blackhole = blackhole
        self._calculate_local_coordinates()

    def _calculate_local_coordinates(self):
        """Calculate local coordinates once and store them as attributes."""                       #| 0.0025
        self._localz = -self.camera.size * (self.pixely / self.camera.resolution - 1/2) - 1 / (2 * self.camera.resolution)#1p = 0.0025, 200p = 1
        self._localy = self.camera.size * (self.pixelx / self.camera.resolution - 1/2) + 1 / (2 * self.camera.resolution)
        self._localx = self.camera.distance
        self._localr = sqrt(self.localx ** 2 + self.localy ** 2 + self.localz ** 2)
        self._localtheta = np.arccos(self.localz / np.sqrt(self.localx ** 2 + self.localz ** 2))
        self._localphi = np.arctan(self.localy / self.localx) + pi

    @property
    def pixelx(self):
        """Returns the x coordinate of the pixel"""
        return self._pixelx

    @property
    def pixely(self):
        """Returns the y coordinate of the pixel"""
        return self._pixely

    @property
    def camera(self):
        """Returns the camera object"""
        return self._camera

    @property
    def background(self):
        """Returns the background object"""
        return self._background

    @property
    def blackhole(self):
        """Returns the blackhole object"""
        return self._blackhole

    @property
    def localz(self):
        """Returns the local z-coordinate"""
        return self._localz

    @property
    def localy(self):
        """Returns the local y-coordinate"""
        return self._localy

    @property
    def localx(self):
        """Returns the local x-coordinate"""
        return self._localx

    @property
    def localr(self):
        """Returns the local radius"""
        return self._localr

    @property
    def localtheta(self):
        """Returns the local theta angle"""
        return self._localtheta

    @property
    def localphi(self):
        """Returns the local phi angle"""
        return self._localphi

    def _dSdl(self, l, s, m):
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

    def _solver(self, t0, r0, theta0, phi0, pt0, pr0, ptheta0, pphi0, l_span, M):
        s_0 = (t0, r0, theta0, phi0, pt0, pr0, ptheta0, pphi0)
        sol = solve(self._dSdl, l_span, s_0, method='DOP853', args=[M])
        return sol

    def get_color(self):
        """Returns the color of the pixel this ray passes through."""
        pr0, ptheta0, pphi0 = -self.localx / (1 - 2 * self.blackhole.mass / self.camera.r), -self.localz * self.camera.r / sqrt(1 - 2 * self.blackhole.mass / self.camera.r), self.localy * self.camera.r * sin(self.camera.theta) / sqrt(1 - 2 * self.blackhole.mass / self.camera.r)
        if ptheta0**2 + pphi0**2/sin(self.camera.theta)**2  <= 27 * self.blackhole.mass**2:
            return 0, 0, 0
        sol = self._solver(0, self.camera.r, self.camera.theta, self.camera.phi, -1, pr0, ptheta0, pphi0, (0, 200), self.blackhole.mass)

        return self.background.get_angle_value(sol["y"][2][-1], (sol["y"][3][-1]))

    def get_plot_data(self):
        """Returns data for plotting the ray path."""
        pr0, ptheta0, pphi0 = -self.localx / (
                    1 - 2 * self.blackhole.mass / self.camera.r), -self.localz * self.camera.r / sqrt(
            1 - 2 * self.blackhole.mass / self.camera.r), self.localy * self.camera.r * sin(self.camera.theta) / sqrt(
            1 - 2 * self.blackhole.mass / self.camera.r)
        sol = self._solver(0, self.camera.r, self.camera.theta, self.camera.phi, -1, pr0, ptheta0, pphi0, (0, 200), self.blackhole.mass)
        return sol


def get_color_pixel(args):
    """Get color of the pixel using ray tracing."""

    pixel, rays, resolution = args
    return rays[(pixel // resolution, pixel % resolution)].get_color()


def mainPicture():

    black_hole = BlackHole(1)
    camera = Camera(30, pi / 2, 0, 1, 401, 1)
    result = Image.new('RGB', (camera.resolution, camera.resolution))
    pixels = result.load()
    background = Background(Image.open("colorgridCorrect.png"))
    rays = {}
    begin = time()
    for x in range(camera.resolution):
        for y in range(camera.resolution):
            rays[(x, y)] = Ray(x, y, camera, background, black_hole)
    print("Creating rays: " + str(time() - begin))


    start = time()
    pixel_values = []
    with Pool(os.cpu_count()-1) as pool:
        for pixel in pool.map(get_color_pixel, [(x, rays, camera.resolution) for x in range(camera.resolution**2)]):
            pixel_values.append(pixel)

    for i in range(len(pixel_values)):
        pixels[(i // camera.resolution, i % camera.resolution)]=pixel_values[i]

    pool.close()
    print("Calculating paths with "+ str(os.cpu_count()-1) + " cores: " + str(time() - start))
    result.show()

    '''fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sol = rays[(27,5)].get_plot_data()["y"]
    R, T, P = sol[1], sol[2], sol[3]
    X = R * np.sin(T) * np.cos(P)
    Y = R * np.sin(T) * np.sin(P)
    Z = R * np.cos(T)
    ax.plot(X, Y, Z)
    ax.set_ylim([-10, 10])
    ax.set_xlim([-10, 10])
    ax.set_zlim([-10, 10])
    plt.show()'''

def mainPlot():
    black_hole = BlackHole(0)
    camera = Camera(10, pi / 2, 0, 1, 4, 1)
    background = Background(Image.open("InterstellarWormhole_Fig10.jpg"))
    rays = {}
    begin = time()
    for i in range(camera.resolution):
        for j in range(camera.resolution):
            rays[(i, j)] = Ray(i, j, camera, background, black_hole)
    print("Creating rays: " + str(time() - begin))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(camera.resolution):
        for j in range(camera.resolution):
            sol = rays[(i,j)].get_plot_data()["y"]
            R, T, P = sol[1], sol[2], sol[3]
            X = R * np.sin(T) * np.cos(P)
            Y = R * np.sin(T) * np.sin(P)
            Z = R * np.cos(T)
            ax.plot(X, Y, Z)

    ax.set_ylim([-10, 10])
    ax.set_xlim([-10, 10])
    ax.set_zlim([-10, 10])
    plt.show()


if __name__ == "__main__":
    mainPicture()
