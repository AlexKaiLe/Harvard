import numpy as np
import glob
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

dnames = glob.glob('**/mocap_clean_eh5/*.npy')
egoh5s = []
for dname in dnames:
    print("loaded:", dname)
    egoh5s.append(np.load(dname, allow_pickle=True))
temp = np.array(egoh5s)
print(temp.shape)
egoh5s = np.array([eh5.reshape((eh5.shape[0], eh5.shape[1]//3, 3)) for eh5 in egoh5s])
print(egoh5s.shape)


connections2 = [[5, 4, 6], [4,17,18], [11,12], [0,2,15,17,16,3,1], [13,9,7,18,8,10,14]]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plotrat = egoh5s[0][2356]
print(plotrat.shape)
for conn in connections2:
    ax.plot(plotrat[conn,0], plotrat[conn,1], plotrat[conn,2])
plt.show()


# """
# Matplotlib Animation Example

# author: Jake Vanderplas
# email: vanderplas@astro.washington.edu
# website: http://jakevdp.github.com
# license: BSD
# Please feel free to use and modify this, but keep the above information. Thanks!
# """

# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import animation

# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure()
# ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
# line, = ax.plot([], [], lw=2)

# # initialization function: plot the background of each frame
# def init():
#     line.set_data([], [])
#     return line,

# # animation function.  This is called sequentially
# def animate(i):
#     x = np.linspace(0, 2, 1000)
#     y = np.sin(2 * np.pi * (x - 0.01 * i))
#     line.set_data(x, y)
#     print(line)
#     return line,

# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=200, interval=20, blit=True)

# # save the animation as an mp4.  This requires ffmpeg or mencoder to be
# # installed.  The extra_args ensure that the x264 codec is used, so that
# # the video can be embedded in html5.  You may need to adjust this for
# # your system: for more information, see
# # http://matplotlib.sourceforge.net/api/animation_api.html
# # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# plt.show()