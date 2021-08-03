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

def getAllDistances(matFile, numb):
    mat = matFile[numb]
    mat = mat.T
    dist = cdist(mat, mat, 'euclidean')
    norm = normalize(dist)
    max_dist = max(dist.flatten())
    return dist, norm, max_dist