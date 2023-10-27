#imports
import numpy as np
import time
import math
import PIL.Image as Image
import tqdm.std as tqdm
import multiprocess.pool as Pool
import matplotlib.pyplot as plt
import scipy

#camera class
class Camera:
    #x, y, z are the carthesian coordinates
    #distance is the distance between the camera and the screen
    #resolution is the resolution of the square screen
    #size is the carthesian size of the screen
    def __init__(self, x:float,y:float, z:float , distance:float, resolution:float, size:float):
        self.x = x
        self.y = y
        self.z = z
        self.distance = distance
        self.resolution = resolution
        self.size = size

    #returns carthesian position of the camera
    def getPosition(self) -> (float, float, float):
        return (self.x, self.y, self.z)

    #returns the distance to the screen
    def getDistance(self) -> float:
        return self.distance

    #returns the resolution of the screen
    def getResolution(self) -> float:
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

    #returns the pixel location on the screen
    def getPixelLocation(self) -> (int, int):
        return (self.pixelx, self.pixely)

    #returns the carthesian location of the pixel this ray goes through
    def getLocation(self) -> (float, float, float):
        return (2*self.pixelx*self.camera.size/self.camera.resolution - self.camera.size, 2*self.pixely*self.camera.size/self.camera.resolution - self.camera.size, self.camera.distance)




def main():
    camera = Camera(0, 0, 0, 1000, 400, 10)
    rays = {}



if __name__ == "__main__":
    main()
