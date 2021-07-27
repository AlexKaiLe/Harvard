from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


temporal = loadmat('/Users/alexle/Desktop/Harvard/Python_files/mat_files/newcombed.mat')
print(temporal.keys())
file_times = temporal['Newcombed']
time_data = file_times.reshape((len(file_times), 3*file_times.shape[2]))
print(time_data.shape)
plt.figure(figsize=(1,1))
plt.imshow(time_data.T)
plt.show()