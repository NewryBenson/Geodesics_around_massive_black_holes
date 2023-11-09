#imports
import numpy as np
import time
import math
import PIL.Image as Image
import tqdm.std as tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as Solve
import scipy

class Background:
    def __init__(self, image: Image):
        self.image = image

    def getPixelValue(self, pixelx, pixely):
        return self.image.load()[pixelx, pixely]

    def getAngleValue(self, longtitude, latitude):
        return self.getPixelValue(math.floor(longtitude/(2*np.pi)*(self.image.size[0])), math.floor(latitude/np.pi*(self.image.size[1])))



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
        self.x = 2*pixelx*camera.size/camera.resolution - camera.size + self.camera.size/(2*self.camera.resolution)
        self.y = 2*pixely*camera.size/camera.resolution - camera.size + self.camera.size/(2*self.camera.resolution)
        self.z = camera.distance
        self.angle = np.arctan(np.sqrt(self.x**2+self.y**2)/self.z)
        self.longtitude = np.arctan(self.x/self.z)+np.pi
        self.latitude = np.arctan(self.y / self.z)+np.pi/2
        self.b = camera.r*np.sin(self.angle)
        self.bcr = np.sqrt(27)*blackhole.mass
        self.r1 = np.cbrt(self.b**2*(self.blackhole.mass + np.sqrt(self.blackhole.mass**2-self.b**2/27))) + np.cbrt(self.b**2*(self.blackhole.mass - np.sqrt(self.blackhole.mass**2-self.b**2/27)))
        self.w1 = self.b/self.r1

    def scatteringAngleFunc(self, w):
        return np.sqrt(1-w**2*(1-2*self.blackhole.mass/self.b * w))

    def getScatteringAngle(self):
        2*scipy.integrate.quad(self.scatteringAngleFunc, 0, self.w1)
    #returns the pixel location on the screen
    def getPixelLocation(self) -> (int, int):
        return (self.pixelx, self.pixely)

    #returns the carthesian location of the pixel this ray goes through
    def getPosition(self) -> (float, float, float):
        return (self.x, self.y, self.z)


    def gelCoordinates(self) -> (float, float):
        return (self.longtitude, self.latitude)

    def getColor(self) -> (int, int, int):
        return self.background.getAngleValue(self.longtitude, self.latitude)




#function that determines deflection angle
def deflection_angle(b, bcr):
    return -np.log(b/bcr -1)+np.log(216*(7-4*np.sqrt(3)))-np.pi

#returns a deflection angle depending on the impact parameter b
def path(ray, m):
    outgoing_angle = deflection_angle(ray[0], np.sqrt(27)*m) - ray[1]
    return outgoing_angle

def pixel_to_xy(pixel, resolution):
    return (pixel//resolution, pixel%resolution)

def get_Color_pixel(pixel, resolution, m, deflects, rays):
    x, y = pixel_to_xy(pixel, resolution)
    if rays[math.floor(np.sqrt(((abs(resolution / 2 - x)-1) ** 2 + (abs(resolution / 2 - y)-1) ** 2) / 2))][0] <= np.sqrt(27) * m:
        return (0, 0, 0)
    else:
        switch_truth = [math.floor(deflects[math.floor(np.sqrt(((abs(resolution / 2 - x)-1) ** 2 + (abs(resolution / 2 - y)-1) ** 2) / 2))] / np.pi) % 2 == 0,
                      x >= resolution / 2,
                      y >= resolution / 2]
        match switch_truth:
            case [True, True, True] | [False, False, False]:
                return (255, 0, 0)
            case [True, False, False] | [False, True, True]:
                return (0, 0, 255)
            case [True, True, False] | [False, False, True]:
                return (0, 255, 0)
            case [True, False, True] | [False, True, False]:
                return (255, 255, 0)
            case _:
                return (0, 0, 0)

def pixel_to_xy(pixel, resolution):
    return (pixel//resolution, pixel%resolution)

def get_Color_pixel(pixel, rays, resolution):
    return rays[pixel_to_xy(pixel, resolution)].getColor()

def main():
    black_hole = BlackHole(1)
    camera = Camera(-10, 0, 0, 1, 400, 1)
    screen = Image.new(mode="RGB", size=(camera.getResolution(), camera.getResolution()))
    background = Background(Image.open("ColorGrid.png"))
    rays={}
    screenPixels = screen.load()

    for x in range(camera.getResolution()):
        for y in range(camera.getResolution()):
            rays[(x, y)] = Ray(x, y, camera, background, black_hole)


    for x in range(camera.getResolution()):
        for y in range(camera.getResolution()):
            screenPixels[x, y] = rays[(x, y)].getColor()

    screen.show()

    start = time.time()
    pool = Pool(8)
    pixel_values = np.array(
        pool.starmap(get_Color_pixel, [(x, rays, camera.resolution) for x in range(camera.resolution ** 2)])).reshape(
        (camera.resolution, camera.resolution, 3)).astype(np.uint8)
    screen = Image.fromarray(pixel_values, 'RGB')
    pool.close()
    print(time.time() - start)
    screen.show()


if __name__ == "__main__":
    main()