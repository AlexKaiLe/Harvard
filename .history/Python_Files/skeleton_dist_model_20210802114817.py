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

def getData(data, numb): 
    cnn_inputs = []
    max_dist_list = []
    max_height_list = []
    max_angle_list = []

    bar = progressbar.ProgressBar()

    for i in bar(range(numb)):
        dist, norm_dist, max_dist = getAllDistances(data, i) 
        height, norm_height, max_height = getAllHeights(data, i)
        angle, norm_angle, max_angle = getAllAngles(data, i)

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
    avg_max_dist = median(max_dist_list)
    avg_max_height = median(max_height_list)
    avg_max_angle = median(max_angle_list)
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

def trainTest(train_data, test_data, train_labels, test_labels, numb_train, numb_test):
    index_test = np.linspace(0, len(test_labels), num = numb_test, endpoint=False).astype(int)
    pre_train_data = train_data
    pre_train_labels = train_labels
    if numb_train != len(train_data):
        index_train = np.linspace(0, len(train_labels), num = numb_train, endpoint=False).astype(int)
        pre_train_data = train_data[index_train]
        pre_train_labels = train_labels[index_train]
    pre_test_data = test_data[index_test]
    pre_test_labels = test_labels[index_test]
    # -------------------------------------------------------

    # Get measurement data for every 3D point
    print("Getting messurements")
    train_data = getData(pre_train_data, numb_train)
    test_data = getData(pre_test_data, numb_test)

    # Flatten the trian labels to fit dimentions of data
    train_labels = pre_train_labels.flatten()[0:(numb_train*20)]-1
    test_labels = pre_test_labels.flatten()[0:(numb_test*20)]-1
    # -------------------------------------------------------

    # Get index where the data is all 0
    print("Removing nans")
    nans_train = np.sort(np.where(~train_data.any(axis=1))[0])[::-1]
    nans_test = np.sort(np.where(~test_data.any(axis=1))[0])[::-1]
    # Turn data into lists
    train_data_new = list(train_data)
    test_data_new = list(test_data)
    train_labels_new = list(train_labels)
    test_labels_new = list(test_labels)
    # Remove the nan values
    bar = progressbar.ProgressBar()
    for i in bar(nans_train):
        train_data_new.pop(i)
        train_labels_new.pop(i)
    bar = progressbar.ProgressBar()
    for i in bar(nans_test):
        test_data_new.pop(i)
        test_labels_new.pop(i)
    # Turn data back into array
    train_data_new = np.array(train_data_new)
    train_labels_new = np.array(train_labels_new)
    test_data_new = np.array(test_data_new)
    test_labels_new = np.array(test_labels_new)
    return train_data_new, test_data_new, train_labels_new, test_labels_new

def saveTestTrain(train_data_new, test_data_new, train_labels_new, test_labels_new):
    np.save(save_folder + 'train_data.npy', np.asarray(train_data_new))
    np.save(save_folder + 'train_labels.npy', np.asarray(train_labels_new))
    np.save(save_folder + 'test_data.npy', np.asarray(test_data_new))
    np.save(save_folder + 'test_labels.npy', np.asarray(test_labels_new))

# Creates the ML training platform to predict rat joints
def ml_traning(train_data, train_labels, test_data, test_labels):
    # DIMENTION CHANGE
    train_data = train_data.reshape(train_data.shape[0], 39)
    test_data = test_data.reshape(test_data.shape[0], 39)
    # train_data = train_data.reshape(train_data.shape[0], 60)
    # test_data = test_data.reshape(test_data.shape[0], 60)
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')

    model = createModel()
    ml_folder = os.path.join(save_folder, "training")
    if not os.path.isdir(ml_folder):
        os.mkdir(ml_folder)
    checkpoint_path = ml_folder + "/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)

    model.fit(x=train_data,y=train_labels, verbose='auto', batch_size=100, epochs=5, validation_data=(test_data, test_labels), callbacks=[cp_callback])
    
    loss, acc = model.evaluate(test_data, test_labels, verbose=1)
    print("Loss:", loss)
    print("Accuracy:", acc*100)
    return model

# Creates the model for the CNN
def createModel():
    model = Sequential()
    model.add(Dense(640, activation= LeakyReLU()))
    model.add(Dense(320, activation= LeakyReLU()))
    model.add(Dense(80, activation= LeakyReLU()))
    model.add(Dense(len(joint_name), activation = "softmax"))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def getStoredModel():
    checkpoint_path = save_folder + "/training/cp.ckpt"
    skeleton_model = createModel()
    skeleton_model.load_weights(checkpoint_path).expect_partial()

    # Re-evaluate the model
    loss, acc = skeleton_model.evaluate(test_data_new, test_labels_new)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    return skeleton_model