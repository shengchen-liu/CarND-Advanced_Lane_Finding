import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
from os import getcwd, chdir

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

def calibrate_camera(verbose=False):
    '''

    :param sample: whether draw and display results
    :return:
    '''

    # Mapping each calibration image to number of checkerboard corners
    # Everything is (9,6) for now

    # List of object points and corners for calibration
    objpoints = [] # 3d points in real world
    imgpoints = []  # 2d points in image plane

    # Make a list of calibration images

    images = glob.glob('camera_cal/*.jpg')

    # Go through all images and find corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, save & draw corners
        if ret == True:
            # Save object points and corresponding corners
            objpoints.append(objp)
            imgpoints.append(corners)

            if verbose:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                cv2.imshow('img',img)
                cv2.waitKey(500)
            print('Found corners for %s' % fname)
        else:
            print('Warning: ret = %s for %s' % (ret, fname))

    if verbose:
        cv2.destroyAllWindows()

    # Calibrate camera and undistort a test image
    img = cv2.imread('test_images/straight_lines2.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    return ret, mtx, dist

def undistort(frame, mtx, dist, verbose=False):
    """
    Undistort a frame given camera matrix and distortion coefficients.
    :param frame: input frame
    :param mtx: camera matrix
    :param dist: distortion coefficients
    :param verbose: if True, show frame before/after distortion correction
    :return: undistorted frame
    """
    frame_undistorted = cv2.undistort(frame, mtx, dist, newCameraMatrix=mtx)

    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[1].imshow(cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB))
        plt.show()

    save_dict = {'mtx': mtx, 'dist': dist}
    with open('calibrate_camera.p', 'wb') as f:
        pickle.dump(save_dict, f)

    return frame_undistorted

