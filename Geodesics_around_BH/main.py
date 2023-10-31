#imports
import numpy as np
import time
import math
import PIL.Image as Image
import tqdm.std as tqdm
import multiprocess.pool as Pool
import matplotlib.pyplot as plt
import scipy

class Background:
    def __init__(self, image: Image):
        self.image = image

    def getPixelValue(self, pixely, pixelx):
        return self.image.load()[pixelx, pixely]

    def getAngleValue(self, theta, phi):
        return self.getPixelValue(math.floor(theta/np.pi*self.image.size[1]), math.floor(phi/(2*np.pi)*self.image.size[0]))



#black hole class
class BlackHole:
    def __init__(self,x:float,y:float, z:float, mass: float):
        self.x = x
        self.y = y
        self.z = z
        self.mass = mass

    # returns carthesian position of the black hole
    def getPosition(self) -> (float, float, float):
        return (self.x, self.y, self.z)

    # returns the mass of the black hole
    def getMass(self) -> float:
        return self.mass

#camera class
class Camera:
    #distance is the distance between the camera and the screen
    #resolution is the resolution of the square screen
    #size is the carthesian size of the screen
    def __init__(self , distance:float, resolution:int, size:float):
        self.distance = distance
        self.resolution = resolution
        self.size = size

    #returns carthesian position of the camera
    def getPosition(self) -> (float, float, float):
        return (0, 0, 0)

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
    def __init__(self, pixely: int, pixelz: int, camera: Camera, background: Background):
        self.pixelz = pixelz
        self.pixely = pixely
        self.camera = camera
        self.background = background
        self.z = -2*pixelz*camera.size/camera.resolution + camera.size - self.camera.size/(2*self.camera.resolution)
        self.y = -2*pixely*camera.size/camera.resolution + camera.size - self.camera.size/(2*self.camera.resolution)
        self.x = camera.distance
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.theta = np.arccos(self.z/self.r)
        self.phi = np.arctan(self.y/self.x)+np.pi
        #self.phi = -np.sign(self.y)*np.arccos(self.x/np.sqrt(self.x**2 + self.y**2))+np.pi



    #returns the pixel location on the screen
    def getPixelLocation(self) -> (int, int):
        return (self.pixely, self.pixelz)

    #returns the carthesian location of the pixel this ray goes through
    def getPosition(self) -> (float, float, float):
        return (self.x, self.y, self.z)

    #returns carthesian coordinate in form: (r, phi, theta) where r is the distance from the camera, phi the angle between the ray and the line between the middel of the screen and the camera. Theta is the rotation around that line.
    def getSphericalPosition(self) -> (float, float, float):
        return (self.r, self.theta, self.phi)

    def getColor(self) -> (int, int, int):
        return self.background.getAngleValue(self.theta, self.phi)

    def getColorGradient(self) -> (int, int, int):
        return (math.floor(self.theta / np.pi * 255), 0, math.floor(self.phi / (2 * np.pi) * 255))





#Define all differential equations (coupled)
def dSdl(l, S, M):
    t, r, theta, phi, pt, pr, ptheta, pphi = S
    return [-1/(1-(2*M/r))*pt,
            (1-(2*M/r))*pr,
            (r**-2)*ptheta,
            (r**-2)*pphi,
            0,
            -(1/2)*( (1-(2*M/r))**-2 * (2*M/(r**2))*pt**2 + (2*M/(r**2))*pr**2 -2*(r**-3)*ptheta**2 -2*(r**-3)*np.sin(theta)**2 * pphi**2),
            np.cos(theta)*(np.sin(theta))**-3 * r**-2 * pphi**2,
            0]

#l_span should be an (l0, lfinal) object
def solver(t0, r0, theta0, phi0, pt0, pr0, ptheta0, pphi0, l_span):
    S_0 = (t0, r0, theta0, phi0, pt0, pr0, ptheta0, pphi0)
    sol = scipy.integrate.solve_ivp(dSdl, l_span, S_0, method='RK45')
    return sol



def main():
    black_hole = BlackHole(0, 0, 10, 1)
    camera = Camera(3, 400, 1)
    screen = Image.new(mode="RGB", size=(camera.getResolution(), camera.getResolution()))
    background = Background(Image.open("ColorGrid.png"))
    rays={}
    screenPixels = screen.load()
    for x in range(camera.getResolution()):
        for y in range(camera.getResolution()):
            rays[(x, y)] = Ray(x, y, camera, background)

    for x in range(camera.getResolution()):
        for y in range(camera.getResolution()):
            screenPixels[x, y] = rays[(x, y)].getColor()

    screen.show()


if __name__ == "__main__":
    main()
