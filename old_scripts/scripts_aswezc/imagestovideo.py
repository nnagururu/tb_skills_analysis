import h5py
import numpy as np
from PIL import Image
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', dest='infile') # pass in a directory
args = parser.parse_args()

# Read hdf5 file and get left images
f = h5py.File(args.infile, 'r')
data = f['data']
l_img = data['l_img']
size = l_img[0].shape[:2]

# Write images to video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('vid.mp4', fourcc, 25, (size[1], size[0]))
for frame in range(len(l_img)):
    video.write(l_img[frame])
video.release()

#  https://github.com/adnanmunawar/rosbag2video/blob/master-adnan/rosbag2timestamps.py