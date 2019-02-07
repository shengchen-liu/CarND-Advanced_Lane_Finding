import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from threshold import combined_thresh
from perspective_transform import perspective_transform
from visualization import *
from fit_line import fit_line, measure_curvature_meters, search_around_poly,calc_vehicle_offset
from line import Line
from moviepy.editor import VideoFileClip
import argparse

# Global variables (just to make the moviepy video annotation work)
with open('calibrate_camera.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']
window_size = 5  # how many frames for line smoothing
left_line = Line(n=window_size)
right_line = Line(n=window_size)
left_curve, right_curve = 0., 0.  # radius of curvature for left and right lanes
left_lane_inds, right_lane_inds = None, None # for calculating curvature

# MoviePy video process will call this function
def process_image(img_in):
	"""
	Process the input image with lane line markings
	Returns annotated image
	"""

	# Undistort, threshold, perspective transform
	undist = cv2.undistort(img_in, mtx, dist, None, mtx)
	img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(undist)
	binary_warped, binary_unwarped, m, m_inv = perspective_transform(img)

	# Perform polynomial fit
	detected = False  # did the fast line fit detect the lines?
	if not detected:
		# Slow line fit
		ret = fit_line(binary_warped)
		left_fit = ret['left_fit']
		right_fit = ret['right_fit']
		nonzerox = ret['nonzerox']
		nonzeroy = ret['nonzeroy']
		left_lane_inds = ret['left_lane_inds']
		right_lane_inds = ret['right_lane_inds']

		# Get moving average of line fit coefficients
		left_fit = left_line.add_fit(left_fit)
		right_fit = right_line.add_fit(right_fit)

		# Calculate curvature
		left_curve, right_curve = measure_curvature_meters(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

		detected = True  # slow line fit always detects the line

	else:  # implies detected == True
		# Fast line fit
		left_fit = left_line.get_fit()
		right_fit = right_line.get_fit()
		ret = search_around_poly(binary_warped, left_fit, right_fit)
		left_fit = ret['left_fit']
		right_fit = ret['right_fit']
		nonzerox = ret['nonzerox']
		nonzeroy = ret['nonzeroy']
		left_lane_inds = ret['left_lane_inds']
		right_lane_inds = ret['right_lane_inds']

		# Only make updates if we detected lines in current frame
		if ret is not None:
			left_fit = ret['left_fit']
			right_fit = ret['right_fit']
			nonzerox = ret['nonzerox']
			nonzeroy = ret['nonzeroy']
			left_lane_inds = ret['left_lane_inds']
			right_lane_inds = ret['right_lane_inds']

			left_fit = left_line.add_fit(left_fit)
			right_fit = right_line.add_fit(right_fit)
			left_curve, right_curve = measure_curvature_meters(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
		else:
			detected = False

	vehicle_offset = calc_vehicle_offset(undist, left_fit, right_fit)

	# Perform final visualization on top of original undistorted image
	result = final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)

	return result

def process_video(input_file, output_file):
	""" Given input_file video, save annotated video to output_file """
	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(process_image)
	annotated_video.write_videofile(output_file, audio=False)

if __name__ == '__main__':
	# -----------------------------------------------
	# Arg parser
	# changes
	parser = argparse.ArgumentParser()
	parser.add_argument("--VIDEO_NAME", help="NAME OF VIDEO FOR TEST", type=str)
	args = parser.parse_args()
	video_name = args.VIDEO_NAME
	video_output = 'test_videos_output/{}.mp4'.format(video_name)

	process_video('{}.mp4'.format(video_name), video_output)

	img_file = 'test_images/test2.jpg'
	img_in = mpimg.imread(img_file)
	result = process_image(img_in)
	plt.imshow(result)
	plt.show()
	print("done")
