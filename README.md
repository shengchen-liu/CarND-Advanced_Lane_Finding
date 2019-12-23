# Advanced Lane Finding Project
### This is Project 2 of Udacity's Self-Driving Car Nanodegree Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image0]: ./examples/Pipeline.png "Pipeline"
[image1]: ./output_images/undistort_calibration.png "Undistorted"
[image2]: ./output_images/test2_undistorted.png "Road Transformed"
[image3]: ./output_images/binary.png "Binary Example"
[image4]: ./output_images/perspective_transform.png "Warp Example"
[image5]: ./output_images/curve_fitting.png "Fit Visual"
[image6]: ./output_images/test2.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[image_challenge]: ./test_videos_output/project_video.gif "challenge"

Python, pycharm

To view the video on Youtube, click the following image:

[![IMAGE ALT TEXT HERE](./test_videos_output/project_video.gif)](https://youtu.be/TVTIUdusZGc) 


The goals / steps of this project are the following:

![alt text][image0]

---
## How to run the code?

This project requires Python3.

Type 'python main.py --MODE=[Mode_Type]  --FILE_NAME=[File_Name] '

For 'Mode_Type' option, choose from 'VIDEO' or 'IMAGE'

For 'File_Name' option, type the name of the source that you want to work with.

---

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file "calibration_utils.py", defined by function "calibrate_camera", and "undistort".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 11 through 79 in `threshold.py`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 1 through 8 in the file `perspective_transform.py` .  The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:
         
| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 300, 7        | 
| 1100, 720     | 980, 720      |
| 595, 450      | 300, 0        |
| 685, 450      | 980, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for my curve fitting includes two functions called `get_fits_by_sliding_windows()`, and `get_fits_by_previous_fits()` which appears in lines 100 through 315 in the file `line_utils.py` .  

I fit my lane lines with a 2nd order polynomial shown as following:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for calculating the radius of curvature includes in a class called `Line` which appears in lines 15 through 98 in the file `line_utils.py` .  

The code for calculating the position of the vehicle with respect to center includes in a function called `compute_offset_from_center` which appears in lines 317 through 340 in the file `line_utils.py` .  


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 139 through 180 in my code in `visualization.py` in the function `prepare_out_blend_frame()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  

Here's a [link to my video result](https://youtu.be/TVTIUdusZGc)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The technique shown works very well to the situation it was design for. For example, it picks the yellow and white lanes, so it will not work well in situations were panes are blue, or pink, for example. Or when the curves are outside the chosen boundary region (and a too broad region will introduce noise). The overtuning of parameters will make it not able to generalize the method.

Computer vision techniques are straigthforward, in comparison to recognition by deep learning for example. By straightforward, I mean we explicitly define the steps we want to take (undistort, detect edges, pick colors, and so on). By the other hand, in deep learning we do not explicitly choose these steps. Deep learning can make the algorithm more robust sometimes, other times make it fail for reasons nobody knows why.

Perhaps the best conclusion to take is that it is easy to create a simple algorithm that performs relatively well, but it is very very hard to create one that will have a human level performance, to handle every situation. There are a lot of improvements to be done. Imagine a lane finding algorithm at night? Or under rain? Perhaps a combination of approachs can make the final result more robust. Or the algorithm can have different tunings for different roads, for example.