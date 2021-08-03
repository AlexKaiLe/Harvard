import numpy as np
import os

import matplotlib.pyplot as plt
from scipy.io import loadmat
import tensorflow as tf
from statistics import median
import progressbar
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist


# global variables
joint_name = ['HeadF', 'HeadB', 'HeadL', 'SpineF', 'SpineM', 'SpineL', 
            'Offset1', 'Offset2', 'HipL', 'HipR', 'ElbowL', 'ArmL', 
            'ShoulderL', 'ShoulderR', 'ElbowR', 'ArmR', 'KneeR', 
            'KneeL', 'ShinL', 'ShinR']

joints_idx = [[1, 2], [2, 3], [1, 3], [2, 4], [1, 4], [3, 4], [4, 5], 
            [5, 6], [4, 7], [7, 8], [5, 8], [5, 7], [6, 8], [6, 9], 
            [6, 10], [11, 12], [4, 13], [4, 14], [11, 13], [12, 13], 
            [14, 15], [14, 16], [15, 16], [9, 18], [10, 17], [18, 19], 
            [17, 20]]

save_folder = 'datasets/skeleton/'
def createSaveFolder():
    if not os.path.isdir('datasets'):
        os.mkdir('datasets')
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

def getAllDistances(matFile, numb):
    mat = matFile[numb]
    mat = mat.T
    dist = cdist(mat, mat, 'euclidean')
    norm = normalize(dist)
    max_dist = max(dist.flatten())
    return dist, norm, max_dist

# Get the absolute height difference to every single point 
def getAllHeights(matFile, numb):
    mat = matFile[numb]
    z = mat[2]
    num_pts = int(matFile.shape[2])
    height_diff = []

    for i in range(num_pts):
        for j in range(num_pts):
            if np.nan in [z[i], z[j]]:
                height_diff.append(np.nan)
            else:
                height_diff.append(np.abs(z[i]-z[j]))

    # normalizes height data
    height = np.array(height_diff).reshape(num_pts, num_pts)
    norm = normalize(height)
    max_height = max(height.flatten())
    return height, norm, max_height

# Get the angle to every single point 
def getAllAngles(matFile, numb):
    mat = matFile[numb]
    mat = mat.T
    angle = cdist(mat, mat, 'cosine')
    norm = normalize(angle)
    max_angle = max(angle.flatten())
    return angle, norm, max_angle