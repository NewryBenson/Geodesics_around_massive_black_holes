#imports
import numpy as np
import time
import PIL.Image as Image
import tqdm.std as tqdm
import multiprocess.pool as Pool
import matplotlib.pyplot as plt
import scipy

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
    def __init__(self, pixelx: int, pixely: int, camera: Camera):
        self.pixelx = pixelx
        self.pixely = pixely
        self.camera = camera
        self.x = 2*pixelx*camera.size/camera.resolution - camera.size
        self.y = 2*pixely*camera.size/camera.resolution - camera.size
        self.z = camera.distance
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.phi = np.arccos(self.z/self.r)
        self.theta = self.y/abs(self.y)*np.arccos(self.x/np.sqrt(self.x**2 + self.y**2))

    #returns the pixel location on the screen
    def getPixelLocation(self) -> (int, int):
        return (self.pixelx, self.pixely)

    #returns the carthesian location of the pixel this ray goes through
    def getPosition(self) -> (float, float, float):
        return (self.x, self.y, self.z)

    #returns carthesian coordinate in form: (r, phi, theta) where r is the distance from the camera, phi the angle between the ray and the line between the middel of the screen and the camera. Theta is the rotation around that line.
    def getSphericalPosition(self) -> (float, float, float):
        return (self.r, self.phi, self.theta)

    def getColor(self) -> (int, int, int):
        return (abs(self.phi/np.pi*255), 0, abs(self.theta/2/np.pi *255))








def main():
    black_hole = BlackHole(0, 0, 10, 1)
    camera = Camera(1, 400, 1)
    screen = Image.new(mode="RGB", size=(camera.getResolution(), camera.getResolution()))
    rays={}
    screenPixels = screen.load()
    for x in range(camera.getResolution()):
        for y in range(camera.getResolution()):
            rays[(x, y)] = Ray(x, y, camera)

    for x in range(camera.getResolution()):
        for y in range(camera.getResolution()):
            screenPixels[x, y] = rays[(x, y)].getColor()

    screen.show()




if __name__ == "__main__":
    main()
