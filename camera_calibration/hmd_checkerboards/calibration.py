import numpy as np
import cv2 as cv
import glob

w = 7
h = 7
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:h,0:w].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpeg')

calib = False
iphone_mtx = np.load('../iphone_checkerboards/iphone_intrinsics.npy')  
iphone_dist = np.load('../iphone_checkerboards/iphone_distortion.npy')

cv.startWindowThread()
for fname in images:
    print(fname)
    if calib: # if we want to undistort the images using the iphone calibration
        img_og = cv.imread(fname)
        h_img, w_img = img_og.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(iphone_mtx, iphone_dist, (w_img,h_img), 1, (w_img,h_img))
        img = cv.undistort(img_og, iphone_mtx, iphone_dist, None, newcameramtx)
        cv.imshow('calibresult.png', img)
        cv.waitKey(0)
    else:
        img = cv.imread(fname)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCornersSB(gray, (h,w), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (h,w), corners2, ret)
        cv.imshow(fname, img)
        cv.waitKey(500)
        # cv.waitKey(0)


cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

np.save("hmd_intrinsics_uncal.npy", mtx)
np.save("hmd_distortion_uncal.npy", dist)