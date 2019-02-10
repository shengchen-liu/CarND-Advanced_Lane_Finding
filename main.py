import cv2
import os
import matplotlib.pyplot as plt
from calibration_utils import calibrate_camera, undistort
from threshold import binarize, binarize_with_threshold
from perspective_transform import perspective_transform
import pickle
import glob
from line_utils import Line, get_fits_by_sliding_windows, get_fits_by_previous_fits, compute_offset_from_center
from globals import *
from visualization import final_viz, prepare_out_blend_frame
import argparse
from moviepy.editor import VideoFileClip

#global variables
processed_frames = 0
line_lt = Line(buffer_len=window_size)  # line on the left of the lane
line_rt = Line(buffer_len=window_size)  # line on the right of the lane

def process_pipeline(frame, keep_state=True):
    """
    Apply whole lane detection pipeline to an input color frame.
    :param frame: input color frame
    :param keep_state: if True, lane-line state is conserved (this permits to average results)
    :return: output blend with detected lane overlaid
    """

    global line_lt, line_rt, processed_frames

    # undistort the image using coefficients found in calibration
    img_undistorted = undistort(frame, mtx, dist, verbose=False)

    # binarize the frame s.t. lane lines are highlighted as much as possible
    # img_binary = binarize(img_undistorted, verbose=False)
    img_binary = binarize_with_threshold(img_undistorted, verbose=False)

    #  perspective transform
    binary_warped, m, m_inv = perspective_transform(img_binary, verbose=False)

    # fit 2-degree polynomial curve onto lane lines found
    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        # Fast line fit
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(binary_warped, line_lt, line_rt, verbose=False)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(binary_warped, line_lt, line_rt, n_windows=9, verbose=False)

    # compute offset in meter from center of the lane
    offset_meter = compute_offset_from_center(line_lt, line_rt, frame_width=frame.shape[1])

    # Perform final visualization on top of original undistorted image
    result = final_viz(img_undistorted, m_inv, line_lt, line_rt, keep_state)

    # stitch on the top of final output images from different steps of the pipeline
    blend_output = prepare_out_blend_frame(result, img_binary, binary_warped, img_fit, line_lt, line_rt, offset_meter)

    processed_frames += 1

    return blend_output

def process_video(input_file, output_file):
    """ Given input_file video, save annotated video to output_file """
    # video = VideoFileClip(input_file).subclip(40,44) # from 38s to 46s
    video = VideoFileClip(input_file)
    annotated_video = video.fl_image(process_pipeline)
    annotated_video.write_videofile(output_file, audio=False)

if __name__ == '__main__':
    if not os.path.exists('calibrate_camera.p'):
        ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')
    with open('calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']

    parser = argparse.ArgumentParser()
    parser.add_argument("--MODE", help="IMAGE OR VIDEO", type=str)
    parser.add_argument("--FILE_NAME", help="NAME OF FILE FOR TEST", type=str)
    args = parser.parse_args()

    mode = args.MODE
    file_name = args.FILE_NAME
    if mode == "VIDEO":
        video_output = 'test_videos_output/{}'.format(file_name)
        process_video(file_name, video_output)


