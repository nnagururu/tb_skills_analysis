import numpy as np
import cv2 as cv

mtx = np.load('hmd_intrinsics_uncal.npy')
dist = np.load('hmd_distortion_uncal.npy')  

img = cv.imread('IMG_5877.jpeg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
cv.imshow('calibresult.png', dst)
cv.waitKey(0)
