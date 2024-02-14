# # import numpy as np

# # a = np.array([[0,1,0],[1,1,1], [0,0,1]])
# # b = np.array([[1,0,0],[1,0,1], [1,0,1]])

# # print((a+b)/2)


# import numpy as np
# from pylab import *
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

# def randrange(n, vmin, vmax):
#     return (vmax-vmin)*np.random.rand(n) + vmin

# fig = plt.figure(figsize=(8,6))

# ax = fig.add_subplot(111,projection='3d')
# n = 100

# xs = randrange(n, 23, 32)
# ys = randrange(n, 0, 100)
# zs = randrange(n, 0, 100)

# colmap = cm.ScalarMappable(cmap=cm.hsv)
# colmap.set_array(zs)

# yg = ax.scatter(xs, ys, zs, c=cm.hsv(zs/max(zs)), marker='o')
# cb = fig.colorbar(colmap)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()

import numpy as np

# Create a sample [N, N, N, 4] array
N = 3
input_array = np.random.rand(N, N, N, 4)
print(input_array)

# Reshape the input array to [N^3, 7] array
N_cube = N**3
reshaped_array = input_array.reshape(N_cube, 7)

# The first three columns represent X, Y, Z dimensions, and the last four columns represent RGBA color.

# Example: Print the reshaped array
print(reshaped_array)