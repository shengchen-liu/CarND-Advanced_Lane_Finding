import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# selected threshold to highlight yellow lines
yellow_HSV_th_min = np.array([0, 100, 100])
yellow_HSV_th_max = np.array([50, 255, 255])

def thresh_frame_sobel(frame, kernel_size):
    """
    Apply Sobel edge detection to an input frame, then threshold the result
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)

    _, sobel_mag = cv2.threshold(sobel_mag, 50, 1, cv2.THRESH_BINARY)

    return sobel_mag.astype(bool)

def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
    """
    Takes an image, gradient orientation, and threshold min/max values
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
    """
    Return the magnitude of the gradient
    for a given sobel kernel size and threshold values
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Return the direction of the gradient
    for a given sobel kernel size and threshold values
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def hls_thresh(img, thresh=(100, 255)):
    """
    Convert RGB to HLS and threshold to binary image using S channel
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def hsv_thresh(img, th_min, th_max, verbose=False):
    """
    Threshold a color frame in HSV space
    """
    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    min_th_ok = np.all(HSV > th_min, axis=2)
    max_th_ok = np.all(HSV < th_max, axis=2)

    out = np.logical_and(min_th_ok, max_th_ok)

    if verbose:
        plt.imshow(out, cmap='gray')
        plt.show()

    return out

def binarize_with_threshold(img, verbose=False):
    # img: RGB
    h, w = img.shape[:2]

    binary = np.zeros(shape=(h, w), dtype=np.uint8)

    # absolute value of gradient

    abs_bin = abs_sobel_thresh(img, orient='x', thresh_min=50, thresh_max=255)

    # magnitude of gradient
    mag_bin = mag_thresh(img, sobel_kernel=3, mag_thresh=(50, 255))

    # direction of the gradient
    dir_bin = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))

    # Convert RGB to HLS and threshold to binary image using S channel
    hls_bin = hls_thresh(img, thresh=(170, 255))

    # highlight white lines by thresholding the equalized frame
    eq_white_mask = get_binary_from_equalized_grayscale(img, verbose=False)

    # highlight yellow lines by threshold in HSV color space
    HSV_yellow_mask = hsv_thresh(img, yellow_HSV_th_min, yellow_HSV_th_max, verbose=False)
    binary[HSV_yellow_mask]=1

    combined = np.zeros_like(dir_bin)
    combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | ((eq_white_mask == 1) | (binary == 1)) ] = 1

    # combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | eq_white_mask == 1] = 1

    # apply a light morphology to "fill the gaps" in the binary image
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(combined.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # return combined, abs_bin, mag_bin, dir_bin, hls_bin  # DEBUG
    if verbose:
        f, ax = plt.subplots(3, 3)
        f.set_facecolor('white')
        ax[0, 0].imshow(img)
        ax[0, 0].set_title('input_frame')
        ax[0, 0].set_axis_off()
        # ax[0, 0].set_axis_bgcolor('red')
        ax[0, 1].imshow(abs_bin, cmap='gray')
        ax[0, 1].set_title('abs_bin')
        ax[0, 1].set_axis_off()

        ax[0, 2].imshow(mag_bin, cmap='gray')
        ax[0, 2].set_title('mag_bin ')
        ax[0, 2].set_axis_off()

        ax[1, 0].imshow(dir_bin, cmap='gray')
        ax[1, 0].set_title('dir_bin')
        ax[1, 0].set_axis_off()

        ax[1, 1].imshow(eq_white_mask, cmap='gray')
        ax[1, 1].set_title('eq_white_mask')
        ax[1, 1].set_axis_off()

        ax[1, 2].imshow(closing, cmap='gray')
        ax[1, 2].set_title('after closure')
        ax[1, 2].set_axis_off()

        ax[2, 0].imshow(HSV_yellow_mask, cmap='gray')
        ax[2, 0].set_title('HSV_yellow_mask')
        ax[2, 0].set_axis_off()
        plt.show()

    return closing

def get_binary_from_equalized_grayscale(frame, verbose):
    """
    Apply histogram equalization to an input frame, threshold it and return the (binary) result.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    eq_global = cv2.equalizeHist(gray)

    _, th = cv2.threshold(eq_global, thresh=250, maxval=255, type=cv2.THRESH_BINARY)

    if verbose:
        plt.imshow(th, cmap='gray')
        plt.show()

    return th


def binarize(img, verbose=False):
    """
    Convert an input frame to a binary image which highlight as most as possible the lane-lines.

    :param img: input color frame
    :param verbose: if True, show intermediate results
    :return: binarized frame
    """
    h, w = img.shape[:2]

    binary = np.zeros(shape=(h, w), dtype=np.uint8)

    # highlight yellow lines by threshold in HSV color space
    HSV_yellow_mask = hsv_thresh(img, yellow_HSV_th_min, yellow_HSV_th_max, verbose=False)
    binary = np.logical_or(binary, HSV_yellow_mask)

    # highlight white lines by thresholding the equalized frame
    eq_white_mask = get_binary_from_equalized_grayscale(img, verbose=False)
    binary = np.logical_or(binary, eq_white_mask)

    # get Sobel binary mask (thresholded gradients)
    sobel_mask = thresh_frame_sobel(img, kernel_size=9)
    binary = np.logical_or(binary, sobel_mask)

    # apply a light morphology to "fill the gaps" in the binary image
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    if verbose:
        f, ax = plt.subplots(2, 3)
        f.set_facecolor('white')
        ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title('input_frame')
        ax[0, 0].set_axis_off()
        # ax[0, 0].set_axis_bgcolor('red')
        ax[0, 1].imshow(eq_white_mask, cmap='gray')
        ax[0, 1].set_title('white mask')
        ax[0, 1].set_axis_off()

        ax[0, 2].imshow(HSV_yellow_mask, cmap='gray')
        ax[0, 2].set_title('yellow mask')
        ax[0, 2].set_axis_off()

        ax[1, 0].imshow(sobel_mask, cmap='gray')
        ax[1, 0].set_title('sobel mask')
        ax[1, 0].set_axis_off()

        ax[1, 1].imshow(binary, cmap='gray')
        ax[1, 1].set_title('before closure')
        ax[1, 1].set_axis_off()

        ax[1, 2].imshow(closing, cmap='gray')
        ax[1, 2].set_title('after closure')
        ax[1, 2].set_axis_off()
        plt.show()

    return closing



if __name__ == '__main__':
    img_file = 'test_images/straight_lines1.jpg'
    img_file = 'test_images/test5.jpg'

    # with open('calibrate_camera.p', 'rb') as f:
    #     save_dict = pickle.load(f)
    # mtx = save_dict['mtx']
    # dist = save_dict['dist']
    #
    # img = mpimg.imread(img_file)
    # img = cv2.undistort(img, mtx, dist, None, mtx)
    img = cv2.imread(img_file)
    closing = binarize(img, verbose=True)
