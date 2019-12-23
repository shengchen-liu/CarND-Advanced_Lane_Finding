import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import os
from os import getcwd, chdir

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

def calibrate_camera(calib_images_dir, verbose=False):
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

    images = glob.glob(calib_images_dir + '/*.jpg')

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

    # save as pickle
    result_dict = {}
    result_dict['mtx'] = mtx
    result_dict['dist'] = dist

    # Its important to use binary mode
    dbfile = open('calibrate_camera.p', 'ab')

    # source, destination
    pickle.dump(result_dict, dbfile)
    dbfile.close()

    return ret, mtx, dist, rvecs, tvecs

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
        ax[0].set_title('Original image')
        ax[1].imshow(cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Undistorted image')
        plt.show()

    if not os.path.exists('calibrate_camera.p'):
        save_dict = {'mtx': mtx, 'dist': dist}
        with open('calibrate_camera.p', 'wb') as f:
            pickle.dump(save_dict, f)

    return frame_undistorted

if __name__ == '__main__':
    if not os.path.exists('calibrate_camera.p'):
        ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal', verbose=True)
    with open('calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']

    # img = cv2.imread('camera_cal/calibration1.jpg')
    # undistorted = undistort(img, mtx, dist, verbose=True)

    img = cv2.imread('test_images/test2.jpg')
    undistorted = undistort(img, mtx, dist, verbose=True)
    cv2.imwrite("output_images/test2_undistorted.png", undistorted)



