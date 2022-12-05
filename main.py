import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
import os
import math

resolution_var = 0

width = [400, 800, 1280, 1920][resolution_var]
height = [300, 600, 720, 1080][resolution_var]

no_frames = 150
fps = 30

free_space_ratio = 1300

initial_velocity_limit = 5
critical_distance = 50
force_factor = -1
force_type = 2
no_adj_birds = 10
drag_coef = 0.8

# np.array([0, 255, 255], np.uint8)
single_bird = np.array([0, 0, 0], np.uint8)
single_sky = np.array([0, 0, 0], np.uint8)

# SKY = np.full((height, width, 3), single_sky[0], dtype=np.uint8)

SKY_IM = cv2.imread('sky_image.png')
SKY = np.array([[SKY_IM[j][i] for i in range(width)] for j in range(height)])
SKY_COPY = np.array([[SKY_IM[j][i] for i in range(width)]
                    for j in range(height)])


# x_coor, y_coor, x_v, y_v, x_force = 0, y_force = 0
STARLINGS = np.array([[], [], [], [], [], []], dtype=np.int32)


total_energy_trend = []


###
###

def force_func(dist):
    x = dist/critical_distance
    if force_type == 0:
        return 0
    elif force_type == 1:
        if x <= 1:
            return force_factor * np.log(x)
        else:
            return force_factor * np.exp(x - 1) - 1
    else:
        return force_factor * x


def force_func_plot(start=0.01):
    x_list = np.arange(start, 2 * critical_distance, 0.001)
    plt.plot(x_list, list(map(force_func, x_list)))
    plt.show()


def total_energy():
    return ((np.array(STARLINGS[2])**2 + np.array(STARLINGS[3])**2).sum())


def energy_plotter():
    x_list = np.array([i for i in range(len(total_energy_trend))])
    plt.plot(x_list, total_energy_trend)
    plt.show()


def total_momentum():
    return [sum(list(map(abs, STARLINGS[2]))), sum(list(map(abs, STARLINGS[3])))]

###
###


def birdmaker():
    global SKY, STARLINGS

    SKY = SKY.reshape((height * width, 3))
    blacky = random.choices([0, 1], k=len(SKY), weights=[free_space_ratio, 1])

    for i, j in enumerate(blacky):
        if j == 1:
            STARLINGS = np.append(STARLINGS, np.array([[i % width], [i // width], [random.randint(-initial_velocity_limit, initial_velocity_limit)], [
                                  random.randint(-initial_velocity_limit, initial_velocity_limit)], [0], [0]]), axis=1)

            SKY[i] = np.array(single_bird, dtype=np.uint8)

    SKY = SKY.reshape((height, width, 3))


birdmaker()

NO_BIRDS = len(STARLINGS[0])


def birdmover():
    for bird_ind in range(NO_BIRDS):
        v_factor = 1

        rebound_xy = [0, 0]

        xini, yini = STARLINGS[0][bird_ind], STARLINGS[1][bird_ind]
        xf, yf = xini + v_factor * \
            STARLINGS[2][bird_ind], yini + v_factor * STARLINGS[3][bird_ind]

        if xf not in range(width):
            rebound_xy[0] = 1
            xf = width - (xf % width)-1
        if yf not in range(height):
            rebound_xy[1] = 1
            yf = height - (yf % height)-1
        if 1:  # SKY[yf, xf].tolist() == SKY_COPY[yf, xf].tolist():
            SKY[yf, xf] = single_bird
            SKY[yini, xini] = SKY_COPY[yini, xini]

            STARLINGS[0][bird_ind], STARLINGS[1][bird_ind] = xf, yf

        if rebound_xy[0] == 1:
            STARLINGS[2][bird_ind] = -STARLINGS[2][bird_ind]
        if rebound_xy[1] == 1:
            STARLINGS[3][bird_ind] = -STARLINGS[3][bird_ind]


def forcer():
    global no_adj_birds, no_adj_birds
    # no_adj_birds = int(NO_BIRDS/18)
    for bird_ind in range(NO_BIRDS):
        xs = STARLINGS[0].copy()
        ys = STARLINGS[1].copy()

        x0, y0 = xs[bird_ind], ys[bird_ind]

        dist_list = [[i, math.sqrt((xs[i] - x0)**2 + (ys[i] - y0)**2)]
                     for i in range(len(xs))]
        dist_list.pop(bird_ind)
        dist_list.sort(key=lambda l: l[1])

        adj_birds_dists = [i for i in dist_list[0:no_adj_birds]]

        # right, down +ve | own - other
        dists_x_y = [[x0 - STARLINGS[0][i], y0 - STARLINGS[1][i]]
                     for i in [j[0] for j in adj_birds_dists]]

        forces = list(map(force_func, [j[1] for j in adj_birds_dists]))

        for i in range(no_adj_birds):
            if adj_birds_dists[i][1]:
                fx, fy = int(forces[i] * dists_x_y[i][0]/adj_birds_dists[i]
                             [1]), int(forces[i] * dists_x_y[i][1]/adj_birds_dists[i][1])
            else:
                fx, fy = 0, 0
            STARLINGS[4][bird_ind] += fx
            STARLINGS[5][bird_ind] += fy

            STARLINGS[4][adj_birds_dists[i][0]] -= fx
            STARLINGS[5][adj_birds_dists[i][0]] -= fy
        STARLINGS[4][bird_ind] -= int(drag_coef * STARLINGS[2][bird_ind])
        STARLINGS[5][bird_ind] -= int(drag_coef * STARLINGS[3][bird_ind])


def forcer_2(bird_ind):
    global no_adj_birds
    no_adj_birds = int(NO_BIRDS/7)
    xs = STARLINGS[0].copy()
    ys = STARLINGS[1].copy()

    x0, y0 = xs[bird_ind], ys[bird_ind]

    dist_list = [[i, math.sqrt((xs[i] - x0)**2 + (ys[i] - y0)**2)]
                 for i in range(len(xs))]
    dist_list.pop(bird_ind)
    dist_list.sort(key=lambda l: l[1])


def accerlerator():
    STARLINGS[2:4] += STARLINGS[4:6]
    STARLINGS[4:6] -= STARLINGS[4:6]


def main():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(total_energy(), total_momentum())
    vid_no_var = 0
    while 1:
        if not os.path.exists(f'starlingssss{vid_no_var}.mp4'):
            out = cv2.VideoWriter(
                f'starlingssss{vid_no_var}.mp4', fourcc, fps, (width, height))
            break
        vid_no_var += 1

    for frame in range(no_frames):
        total_energy_trend.append(total_energy())
        print(f'\r{frame+1}/{no_frames}\t{total_energy()}', end='\t')
        forcer()

        accerlerator()
        if sum(STARLINGS[4]) or sum(STARLINGS[5]):
            print('haha', sum(STARLINGS[4]), sum(STARLINGS[5]))

        birdmover()

        out.write(SKY)
    print('')

    print(total_energy(), total_momentum())


main()


# force_func_plot(1)
energy_plotter()
print('\nDone!')

'''
continuous boundaries
'''
