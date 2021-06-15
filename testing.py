import numpy as np
from numpy.core.numeric import True_
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

# Dataset
def plotHist(x, y):
    print(x, y)
    print("gello")
    x = np.array(x)
    y = np.array(y)
    
    X_Y_Spline = make_interp_spline(x, y)
    
    # Returns evenly spaced numbers
    # over a specified interval.
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = X_Y_Spline(X_)
    
    # Plotting the G
    # raph
    plt.plot(X_, Y_, color='black')
    fig = plt.figure(frameon=False)
    ax= fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor("w")
    # ax = plt.Axes(fig, [0, 0., 1., 1.])
    # ax.set_axis_off()
    plt.fill_between(X_, Y_, color='black')
    plt.savefig("squares1.png", dpi = 200)
    plt.show()
