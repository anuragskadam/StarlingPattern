import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
import os
import math

resolution_var = 0

depth = 400
height = [300, 600, 720, 1080][resolution_var]
width = [400, 800, 1280, 1920][resolution_var]

no_frames = 150
fps = 30

free_space_ratio = 1300
initial_velocity_limit = 10

SKY_3D = np.full((depth, height[resolution_var], height[resolution_var]), 0, np.uint8)
PROJECTION = np.full((height[resolution_var], height[resolution_var], 3), 0, np.uint8)

STARLINGS = np.array([[],[],[],[],[],[],[],[],[]], dtype=)          # z, y, x, vz, vy, vx, fz, fy, fx

single_bird = single_bird = np.array([0, 0, 0], np.uint8)



def bird_maker():
    global SKY_3D, SKY, STARLINGS

    SKY = SKY.reshape((height * width * depth, 3))
    bird_mask = random.choices([0, 1], k=len(SKY), weights=[free_space_ratio, 1])

    for i, j in enumerate(bird_mask):
        if j == 1:
            STARLINGS = np.append(STARLINGS, np.array([[i % width], [i // width], [random.randint(-initial_velocity_limit, initial_velocity_limit)], [
                                  random.randint(-initial_velocity_limit, initial_velocity_limit)], [0], [0]]), axis=1)

            SKY[i] = np.array(single_bird, dtype=np.uint8)

    SKY = SKY.reshape((height, width, 3))

def forcer():
    pass

def accelerator():
    pass

def mover():
    pass

def main():
    pass

main()

