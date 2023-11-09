#imports
import numpy as np
import time
import math
import PIL.Image as Image
import tqdm.std as tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp as Solve
import warnings
warnings.filterwarnings("ignore")

class Background:
    def __init__(self, image: Image):
        self.image = image

    def getPixelValue(self, pixely, pixelx):
        return self.image.load()[pixelx, pixely]

    def getAngleValue(self, theta, phi):
        return self.getPixelValue(math.floor(theta/np.pi*self.image.size[1]), math.floor(phi/(2*np.pi)*self.image.size[0]))



#black hole class
class BlackHole:
    def __init__(self, mass: float):
        self.mass = mass

    # returns carthesian position of the black hole
    def getPosition(self) -> (float, float, float):
        return (0, 0, 0)


    # returns the mass of the black hole
    def getMass(self) -> float:
        return self.mass

#camera class
class Camera:
    #distance is the distance between the camera and the screen
    #resolution is the resolution of the square screen
    #size is the carthesian size of the screen
    def __init__(self,r:float,theta:float, phi:float , distance:float, resolution:int, size:float):
        self.r = r
        self.theta = theta
        self.phi = phi
        self.distance = distance
        self.resolution = resolution
        self.size = size

    #returns carthesian position of the camera
    def getPosition(self) -> (float, float, float):
        return (self.r, self.theta, self.phi)

    #returns the distance to the screen
    def getDistance(self) -> float:
        return self.distance

    #returns the resolution of the screen
    def getResolution(self) -> int:
        return self.resolution

    #returns the size of the screen
    def getSize(self) -> float:
        return self.size

#the ray class
class Ray:
    #pixelx and pixely is the pixel location on the screen of the pixel this ray passes through
    #camera is the camera object the ray comes from
    def __init__(self, pixelx: int, pixely: int, camera: Camera, background: Background, blackhole: BlackHole):
        self.pixelx = pixelx
        self.pixely = pixely
        self.camera = camera
        self.background = background
        self.blackhole = blackhole
        self.localz = -2*pixely*camera.size/camera.resolution + camera.size - self.camera.size/(2*self.camera.resolution)
        self.localy = 2*pixelx*camera.size/camera.resolution - camera.size + self.camera.size/(2*self.camera.resolution)
        self.localx = camera.distance
        self.localr = np.sqrt(self.localx**2 + self.localy**2 + self.localz**2)
        self.localtheta = np.arccos(self.localz/self.localr)
        self.localphi = np.arctan(self.localy/self.localx)+np.pi



    #returns the pixel location on the screen
    def getPixelLocation(self) -> (int, int):
        return (self.pixelx, self.pixely)

    #returns the carthesian location of the pixel this ray goes through
    def getLocalPosition(self) -> (float, float, float):
        return (self.localx, self.localy, self.localz)

    #returns carthesian coordinate in form: (r, phi, theta) where r is the distance from the camera, phi the angle between the ray and the line between the middel of the screen and the camera. Theta is the rotation around that line.
    def getLocalSphericalPosition(self) -> (float, float, float):
        return (self.localr, self.localtheta, self.localphi)


    def getColor(self) -> (int, int, int):
        sol = solver(0, self.camera.r, self.camera.theta, self.camera.phi, -1, -self.localx/(1-2*self.blackhole.mass/self.camera.r), -self.localz*self.camera.r/np.sqrt(1-2*self.blackhole.mass/self.camera.r), -self.localy*self.camera.r*np.sin(self.camera.theta)/np.sqrt(1-2*self.blackhole.mass/self.camera.r), (0, 5000), self.blackhole.mass)
        if sol["y"][1][-1] < 4*self.blackhole.mass:
            return (0, 0, 0)
        return self.background.getAngleValue(sol["y"][2][-1]%(np.pi), sol["y"][3][-1]%(2*np.pi))

    def getPlotData(self):
        sol = solver(0, self.camera.r, self.camera.theta, self.camera.phi, -1,
                     self.localx / (1 - 2 * self.blackhole.mass / self.camera.r),
                     -self.localz * self.camera.r / np.sqrt(1 - 2 * self.blackhole.mass / self.camera.r),
                     -self.localy * self.camera.r * np.sin(self.camera.theta) / np.sqrt(
                         1 - 2 * self.blackhole.mass / self.camera.r), (0, 200), self.blackhole.mass)
        return sol


#Define all differential equations (coupled)
def dSdl(l, S, M):
    t, r, theta, phi, pt, pr, ptheta, pphi = S
    return [-1/(1-(2*M/r))*pt,
            (1-(2*M/r))*pr,
            1/(r**2)*ptheta,
            1/((r**2)*np.sin(theta)**2)*pphi,
            0,
            -(1/2)*(1/((1-(2*M/r))**2) * (2*M/(r**2))*pt**2 + (2*M/(r**2))*pr**2 -2/(r**3)*ptheta**2 -2/((r**3)*np.sin(theta)**2) * pphi**2),
            np.cos(theta)/((np.sin(theta)**3) * (r**2)) * pphi**2,
            0]

#l_span should be an (l0, lfinal) object
def solver(t0, r0, theta0, phi0, pt0, pr0, ptheta0, pphi0, l_span, M):
    S_0 = (t0, r0, theta0, phi0, pt0, pr0, ptheta0, pphi0)
    sol = Solve(dSdl, l_span, S_0, method='RK45', args=[M])
    return sol

def pixel_to_xy(pixel, resolution):
    return (pixel//resolution, pixel%resolution)

def get_Color_pixel(pixel, rays, resolution):
    return rays[pixel_to_xy(pixel, resolution)].getColor()

def main():
    black_hole = BlackHole(1)
    camera = Camera(10, np.pi/2, 0, 1, 100, 1)
    screen = Image.new(mode="RGB", size=(camera.getResolution(), camera.getResolution()))
    background = Background(Image.open("InterstellarWormhole_Fig10.jpg"))
    rays={}
    screenPixels = screen.load()

    begin = time.time()
    for x in range(camera.getResolution()):
        for y in range(camera.getResolution()):
            rays[(x, y)] = Ray(x, y, camera, background, black_hole)
    print("Creating rays: "+ str(time.time()-begin))

    #plotting rays
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for y in range(camera.getResolution()):
        sol = rays[(19,y)].getPlotData()["y"]
        R, T, P = sol[1], sol[2], sol[3]
        X, Y, Z = R * np.sin(T) * np.cos(P), R * np.sin(T) * np.sin(P), R * np.cos(P)
        ax.plot(X, Y, Z)
    plt.show()
    '''


    '''
    begin = time.time()
    for x in range(camera.getResolution()):
        for y in range(camera.getResolution()):
            screenPixels[x, y] = rays[(x, y)].getColor()
    print("Calculating paths with 1 core: " + str(time.time() - begin))
    screen.show()
    '''

    start = time.time()
    pool = Pool(8)
    pixel_values = np.array(
        pool.starmap(get_Color_pixel, [(x, rays, camera.resolution) for x in range(camera.resolution ** 2)])).reshape(
        (camera.resolution, camera.resolution, 3)).astype(np.uint8)
    screen = Image.fromarray(pixel_values, 'RGB')
    pool.close()
    print("Calculating paths with 8 cores: "+ str(time.time() - start))
    screen.show()



if __name__ == "__main__":
    main()
