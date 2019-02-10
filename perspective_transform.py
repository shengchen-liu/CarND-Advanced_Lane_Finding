import numpy as np
import cv2
import matplotlib.pyplot as plt
from calibration_utils import calibrate_camera, undistort
import glob
import matplotlib.image as mpimg
import pickle
from threshold import binarize


def perspective_transform(img, verbose=False):
    """
    Execute perspective transform
    """
    img_size = (img.shape[1], img.shape[0])

    # algorithm to automatically pick?
    # https: // knowledge.udacity.com / questions / 22331
    src = np.float32(
        [[200, 720],
         [1100, 720],
         [595, 450],
         [685, 450]])
    dst = np.float32(
        [[300, 720],
         [980, 720],
         [300, 0],
         [980, 0]])

    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG

    if verbose:
        f, axarray = plt.subplots(1, 2)
        f.set_facecolor('white')
        axarray[0].set_title('Before perspective transform')
        axarray[0].imshow(img, cmap='gray')
        for point in src:
            axarray[0].plot(*point, '.')
        axarray[1].set_title('After perspective transform')
        axarray[1].imshow(warped, cmap='gray')
        for point in dst:
            axarray[1].plot(*point, '.')
        for axis in axarray:
            axis.set_axis_off()
        plt.show()

    return warped, m, m_inv



if __name__ == '__main__':
    with open('calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']

    # show result on test images
    for test_img in glob.glob('test_images/*.jpg'):
        img = cv2.imread(test_img)

        img_undistorted = undistort(img, mtx, dist, verbose=False)

        img_binary = binarize(img_undistorted, verbose=False)

        img_birdeye, M, Minv = perspective_transform(cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB), verbose=True)
