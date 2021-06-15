import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from PIL import Image
import math

joint_name = ['HeadF', 'HeadB', 'HeadL', 'SpineF', 'SpineM', 'SpineL', 
            'Offset1', 'Offset2', 'HipL', 'HipR', 'ElbowL', 'ArmL', 
            'ShoulderL', 'ShoulderR', 'ElbowR', 'ArmR', 'KneeR', 
            'KneeL', 'ShinL', 'ShinR']

joints_idx = [[1, 2], [2, 3], [1, 3], [2, 4], [1, 4], [3, 4], [4, 5], 
            [5, 6], [4, 7], [7, 8], [5, 8], [5, 7], [6, 8], [6, 9], 
            [6, 10], [11, 12], [4, 13], [4, 14], [11, 13], [12, 13], 
            [14, 15], [14, 16], [15, 16], [9, 18], [10, 17], [18, 19], 
            [17, 20]]

def loadMatFile(fileName):
    mat = loadmat('trainTestSplit.mat')[fileName]
    print("loading:", fileName)
    return mat

def plotKnownOrder(matFile, img_numb):
    mat = matFile[img_numb]
    x = mat[0]
    y = mat[1]
    z = mat[2]
    # loading plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # adding points, labels and lines
    try:
        ax.scatter(x,y,z, color='#ff5e5e', s =10, marker='x')
    except:
        pass
    addLabels(ax, x, y, z)
    drawLines(ax, x, y, z)
    dist = drawAllLines(ax, x, y, z)

    # Labeling plot
    ax.set_title("Rat positioning")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.show()
    return dist

def addLabels(ax, x, y, z):
    points_array = []
    # run through all the points to add labels
    for i in range(len(x)):
        try:
            points_array.append([x[i], y[i], z[i]])
            # label = np.around(points_array[i]).astype(int)
            # label = joint_name[i]
            label = ""
            ax.text(x[i], y[i], z[i], label)
            # print(i, ":", points_array[i], joint_name[i])
        except:
            pass
    return points_array

def drawLines(ax,x, y, z):
    # run through all the connections to draw the points
    for i in range(len(joints_idx)):
        try: 
            # Getting both points to draw line
            idx = joints_idx[i]
            x_line = [x[idx[0]-1], x[idx[1]-1]]
            y_line = [y[idx[0]-1], y[idx[1]-1]]
            z_line = [z[idx[0]-1], z[idx[1]-1]]
            z_coord_1 = x[idx[0]-1], y[idx[0]-1], z[idx[0]-1]
            z_coord_2 = x[idx[1]-1], y[idx[1]-1], z[idx[1]-1]

            # Draw lines
            if i < 3: 
                ax.plot(x_line, y_line, z_line, c="#064ea1", linewidth=4)
            elif i < 6:
                ax.plot(x_line, y_line, z_line, c="#64ccd1", linewidth=4)
            else:
                ax.plot(x_line, y_line, z_line, c="#46b8a7", linewidth=4)
        except:
            pass

def drawAllLines(ax,x, y, z):
    joint_len = len(joint_name)
    all_lines = []
    distance = []

    for i in range(joint_len):
        point_dist = []
        for j in range(joint_len):
            point_dist.append(measureDistance(x[i], y[i], z[i], x[j], y[j], z[j]))
            if (not([i,j] in all_lines)):
                x_line = [x[i], x[j]]
                y_line = [y[i], y[j]]
                z_line = [z[i], z[j]]
                ax.plot(x_line, y_line, z_line, color='#b1d8fc', linewidth=0.5)
                all_lines.append([i,j])
                all_lines.append([j,i])
        distance.append(point_dist)
    return distance

def getDistance(mat):
    joint_len = len(joint_name)
    x = mat[0]
    y = mat[1]
    z = mat[2]
    distance = []
    for i in range(joint_len):
        point_dist = []
        for j in range(joint_len):
            point_dist.append(measureDistance(x[i], y[i], z[i], x[j], y[j], z[j]))
        distance.append(point_dist)
    return distance

# def getPointDistance(mat):
#     joint_len = len(joint_name)
#     x = mat[0]
#     y = mat[1]
#     z = mat[2]
#     distance = []
#     for i in range(joint_len):
#         point_dist = []
#         for j in range(joint_len):
#             point_dist.append(measureDistance(x[i], y[i], z[i], x[j], y[j], z[j]))
#         distance.append(point_dist)
#     return distance


def measureDistance(x_1, y_1, z_1, x_2, y_2, z_2):
    x = (x_1 - x_2)
    y = (y_1 - y_2)
    z = (z_1 - z_2)
    return np.sqrt(x**2 + y**2 + z**2)

def heatMap(dist):
    plt.imshow(dist) 
    plt.colorbar()
    plt.grid(False)
    plt.show()

def getAngles(matFile, numb):
    mat = matFile[numb]
    x = np.array(mat[0])
    y = np.array(mat[1])
    z = np.array(mat[2])
    # print(x, y, z)
    angles = []
    for i in range(len(x)):
        curr_point = []
        for j in range(len(y)):
            for k in range(len(z)):
                a = [x[i], y[i], z[i]]
                b = [x[j], y[j], z[j]]
                c = [x[k], y[k], z[k]]
                if not ((a == b) or (b == c) or (c == a)):
                    a = np.array(a)
                    b = np.array(b)
                    c = np.array(c)
                    ba = a - b
                    bc = c - b
                    # print(ba, bc)
                    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle = np.arccos(cosine)
                    angle = np.degrees(angle)
                    curr_point.append(angle)
                else:
                    curr_point.append(0)
        angles.append(curr_point)
    angles = np.array(angles)
    angles = angles.reshape(20, 20, 20)
    print(angles.shape)
    return(angles)

def angleHeatMap(mat_file, numb):
    angles = getAngles(mat_file, numb)
    for i in range(len(angles)):
        heatMap(angles[i])

def getDistData (fileName, numb):
    mat = loadMatFile(fileName)
    training = []
    for i in range(numb):
        dist = np.array(getDistance(mat[i]))
        for i in dist:
            temp = []
            norm = distNorm(i)
            temp.append(norm)
            training.append(temp)
    training = np.array(training)
    return training

def getLabelData(numb):
    output = []
    for i in range(numb):
        for j in range(len(joint_name)):
            output.append(j)
    output = np.array(output)
    return output

def distNorm(point):
    norm = [float(i)/max(point) for i in point]
    return norm

def distHist(dist):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    labels = []
    for i in range(len(dist)):
        labels.append(i)
    ax.bar(labels,dist)
    plt.show()

def distPlot(dist):
    for i in range(len(dist)-1):
        plt.plot([i,i+1], [dist[i],dist[i+1]], 'ro-')
    plt.show()
    
def distBin(dist, numb):
    indicies = []
    output = []
    bins = [0, 0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for i in range(len(dist)):
        # distHist(dist[i][0])
        norm_dist = distNorm(dist[i][0])
        temp = np.ndarray.tolist(np.digitize(norm_dist, bins))
        indicies.append(temp)
    # print(indicies)
    for i in range(numb):
        distHist(count(indicies[i], bins))
        distPlot(count(indicies[i], bins))

def count(indicies, bins):
    output = np.array(np.zeros(11))
    for i in range(10):
        output[i] = indicies.count(i+1)
    return output