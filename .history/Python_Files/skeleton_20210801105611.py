# import statements
import numpy as np
# from numpy import save
# from numpy import asarray
# from numpy import load
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
import tensorflow as tf
import statistics 
import progressbar
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
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

save_folder = 'datasets/data_5/'

# normalizes a matrix
def normalize(matrix):
    max_numb = max(matrix[~np.isnan(matrix)])
    norm = matrix/max_numb
    return np.array(norm)

def getAllDistances(matFile, numb):
    mat = matFile[numb]
    mat = mat.T
    dist = cdist(mat, mat, 'euclidean')
    # norm = normalize(dist)
    # max_dist = max(dist.flatten())
    return dist

# Get the absolute height difference to every single point 
def getAllHeights(matFile, numb):
    mat = matFile[numb]
    z = mat[2]
    reshaped = int(matFile.shape[2])
    height = []

    for i in range(reshaped):
        for j in range(reshaped):
            if np.nan in [z[i], z[j]]:
                height.append(np.nan)
            else:
                height.append(np.abs(z[i]-z[j]))

    # normalizes height data
    matrix = np.array(height).reshape(reshaped, reshaped)
    norm = normalize(matrix)
    max_height = max(norm.flatten())
    return matrix

# Get the angle to every single point 
def getAllAngles(matFile, numb):
    mat = matFile[numb].T
    angle = cdist(mat, mat, 'cosine')
    norm = normalize(angle)
    max_angle = max(angle.flatten())
    return angle

def getData(data, numb): 
    cnn_inputs = []
    max_dist_list = []
    max_height_list = []
    max_angle_list = []

    bar = progressbar.ProgressBar()

    for i in bar(range(numb)):
        dist = getAllDistances(data, i) 
        height = getAllHeights(data, i)
        angle = getAllAngles(data, i)

        # max_dist_list.append(max_dist)
        # max_height_list.append(max_height)
        # max_angle_list.append(max_angle)

        for j in range(0, data.shape[2]):
            temp = np.array([dist[j], height[j], angle[j]])
            first = temp[:,0:3]
            second = temp[:,3:20]
            first = first [ :, first[0].argsort()]
            second = second [ :, second[0].argsort()]
            output = np.concatenate((first, second), axis =1)
            cnn_inputs.append(output)

            big = output[:,:13]
            where_are_NaNs = np.isnan(big)
            big[where_are_NaNs] = 0
            max_dist_list.append(np.max(big[0]))
            max_height_list.append(np.max(big[1]))
            max_angle_list.append(np.max(big[2]))
# median 
    avg_max_dist = statistics.median(max_dist_list)
    avg_max_height = statistics.median(max_height_list)
    avg_max_angle = statistics.median(max_angle_list)
    avg_max = [avg_max_dist, avg_max_height, avg_max_angle]

    cnn_inputs = np.array(cnn_inputs)[:,:,:13]
    final = []
    for i in range(len(cnn_inputs)):
        for j in range(3):
            final.append(cnn_inputs[i][j]/avg_max[j])
    
    cnn_inputs = np.array(final).reshape((cnn_inputs.shape[0], 39))
    where_are_NaNs = np.isnan(cnn_inputs)
    cnn_inputs[where_are_NaNs] = 0
    return cnn_inputs