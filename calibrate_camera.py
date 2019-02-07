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

def calibrate_camera(sample=False):
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

            if sample == True:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
            print('Found corners for %s' % fname)
        else:
            print('Warning: ret = %s for %s' % (ret, fname))

    if sample == True:
        cv2.destroyAllWindows()

    # Calibrate camera and undistort a test image
    img = cv2.imread('test_images/straight_lines2.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    return mtx, dist


if __name__ == '__main__':
    mtx, dist = calibrate_camera(sample=False)
    save_dict = {'mtx': mtx, 'dist': dist}
    with open('calibrate_camera.p', 'wb') as f:
        pickle.dump(save_dict, f)

    # Undistort example calibration image
    img = mpimg.imread('camera_cal/calibration5.jpg')
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    plt.imshow(dst)
    plt.savefig('examples/undistort_calibration.png')