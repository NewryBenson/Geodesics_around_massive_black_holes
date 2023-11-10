import os

import numpy as np
from numpy import pi, sqrt, sin, cos, floor
from time import time
import PIL.Image as Image
from multiprocessing import Pool
from scipy.integrate import solve_ivp as solve
from warnings import filterwarnings
import tqdm

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
        return self.image.load()[floor(phi / (2 * pi) * self.image.size[0]), floor(theta / pi * self.image.size[1])]


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
        """Calculate local coordinates once and store them as attributes."""
        self._localz = -self.camera.size * (2 * self.pixely / self.camera.resolution - 1 + 1 / (2 * self.camera.resolution))
        self._localy = self.camera.size * (2 * self.pixelx / self.camera.resolution - 1 + 1 / (2 * self.camera.resolution))
        self._localx = self.camera.distance
        self._localr = sqrt(self.localx ** 2 + self.localy ** 2 + self.localz ** 2)
        self._localtheta = np.arccos(self.localz / self.localr)
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
        sol = solve(self._dSdl, l_span, s_0, method='RK45', args=[M])
        return sol

    def get_color(self):
        """Returns the color of the pixel this ray passes through."""
        sol = self._solver(0, self.camera.r, self.camera.theta, self.camera.phi, -1,
                           -self.localx / (1 - 2 * self.blackhole.mass / self.camera.r),
                           -self.localz * self.camera.r / sqrt(1 - 2 * self.blackhole.mass / self.camera.r),
                           -self.localy * self.camera.r * sin(self.camera.theta) / sqrt(
                               1 - 2 * self.blackhole.mass / self.camera.r), (0, 200), self.blackhole.mass)
        if sol["y"][1][-1] < 4 * self.blackhole.mass:
            return 0, 0, 0
        return self.background.get_angle_value(sol["y"][2][-1] % pi, sol["y"][3][-1] % (2 * pi))

    def get_plot_data(self):
        """Returns data for plotting the ray path."""
        sol = self._solver(0, self.camera.r, self.camera.theta, self.camera.phi, -1,
                           self.localx / (1 - 2 * self.blackhole.mass / self.camera.r),
                           -self.localz * self.camera.r / sqrt(1 - 2 * self.blackhole.mass / self.camera.r),
                           -self.localy * self.camera.r * sin(self.camera.theta) / sqrt(
                               1 - 2 * self.blackhole.mass / self.camera.r), (0, 200), self.blackhole.mass)
        return sol


def get_color_pixel(args):
    """Get color of the pixel using ray tracing."""
    pixel, rays, resolution = args
    return rays[(pixel // resolution, pixel % resolution)].get_color()


def main():
    black_hole = BlackHole(1)
    camera = Camera(10, pi / 2, 0, 1, 100, 1)
    background = Background(Image.open("InterstellarWormhole_Fig10.jpg"))
    rays = {}
    begin = time()
    for x in range(camera.resolution):
        for y in range(camera.resolution):
            rays[(x, y)] = Ray(x, y, camera, background, black_hole)
    print("Creating rays: " + str(time() - begin))

    start = time()
    pixel_values = []
    with Pool(os.cpu_count()) as pool:
        for pixel in pool.map(get_color_pixel, [(x, rays, camera.resolution) for x in range(camera.resolution**2)]):
            pixel_values.append(pixel)
        pixel_values = np.array(pixel_values).reshape((camera.resolution, camera.resolution, 3)).astype(np.uint8)

    screen = Image.fromarray(pixel_values, 'RGB')
    pool.close()
    print("Calculating paths with 8 cores: " + str(time() - start))
    screen.show()


if __name__ == "__main__":
    main()
