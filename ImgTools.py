# Copyright (c) 2017, Gerti Tuzi
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Gerti Tuzi nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#####################################################################################


import numpy as np
import cv2


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  

    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in np.array(line).reshape(1, 4):
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def binary_mask(C, thresh=(0, 255)):
    """
        Apply threshold to single channel image C
        Binary mask will have values of 1 in the range
        [thresh[0], thresh[1]]
    :param C: image to apply thresholding to.
    :param thresh: 
    :return: 
    binary_mask of the same size and np.uint8 datatype
    """

    binary_output = np.zeros(shape=C.shape, dtype=np.uint8)
    binary_output[(C >= thresh[0]) & (C <= thresh[1])] = 1

    return binary_output

def combine_binary_masks(masks, mode='AND'):
    """
    Combine list of binary mask in a particular mode.
    Expects masks to be the same size. Creates and returns
    a single binary mask 
    :param masks: list of masks to combine
    :param mode: logical operator 
    :return: 
    binary_output
    """
    binary_output = np.zeros(masks[0].shape, dtype=np.uint8)

    for m in masks:
        if mode == 'AND':
            binary_output &=m
        elif mode == 'OR':
            binary_output |=m

    return binary_output

def hls_thresh(img, thresh=(0, 255), channel='S'):

    """
        Define a function that thresholds a channel in HLS
        Use exclusive lower bound (>) and inclusive upper (<=)
        
        Expects a color image in BGR (cv2.imread)
        Default channels is 'S'
        
    :param img: BGR (cv2.imread()) image
    :param thresh: tuple of lower and upper threshold values, 0-255
    :return: 
    Binary mask satisfying the threshold values
    """

    # 0) Determine channel being thresholded
    chanidx = 2
    if channel == 'S':
        pass
    elif channel == 'L':
        chanidx = 1
    elif channel == 'H':
        chanidx = 0

    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    C = hls[:, :, chanidx]

    # 2) Apply binary mask within threshold
    binary_output = binary_mask(C, thresh=thresh)

    # 3) Return a binary image of threshold result
    return binary_output

def bgr_thresh(img, thresh=(0, 255), channel = 'B'):
    """
    Threshold BGR image along a particular channel.
    Function generates a binary image (uint8) with 
    values of 1 in pixels where the magnitude in the 
    selected channel is in the range: [thresh[0], thresh[1]]
    :param img: Image to operate on
    :param thresh: threshold tuple / pair
    :param channel: channel to threshold along. default 'B'
    :return: 
    binary_mask of size == img.shape
    """

    chanidx = 0

    if channel == 'B':
        pass
    elif channel == 'G':
        chanidx = 1
    elif channel == 'R':
        chanidx = 2

    C = img[:, :, chanidx]

    # 2) Apply binary mask within threshold
    binary_output = binary_mask(C, thresh=thresh)

    # 3) Return a binary image of threshold result
    return binary_output

def dir_sobel_thresh(img, orient='x', mag_thresh=(0, 255), sobel_kernel=3):
    '''
    Directional Sobel:
    Take sobel (gradient) of an image with respect to a particular orientation.
    output is an array of the same size as the input image. 
    
    The output array elements are 1 
    where gradients were in the threshold range, 
    and 0 everywhere else
    :param img: input image
    :param orient: direction of the gradient (sobel operator: Sx, Sy, Sx,Sy)
    :param thresh_min: minimum threshold (0-255)
    :param thresh_max: maximum threshold (0-255)
    :return: 
    Binary thresholded image of sobel operator
    '''

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply cv2.Sobel()
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the output from cv2.Sobel()
    abs_sobel = np.absolute(sobel)

    # Scale the result to an 8-bit range (0-255)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Apply lower and upper thresholds
    binary_output = binary_mask(C=scaled_sobel, thresh=mag_thresh)

    # Create binary_output
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Applies Sobel x and y, then computes the magnitude of the gradient
    and applies a threshold

    :param img: image to apply sobel operation on
    :param sobel_kernel: kernel size of sobel. must be an odd number
    :param mag_thresh: tuple of (low thresh, upper thresh)
    :return: 
    binary mask, where entries of 1 indicate location where sobel operation
    was within the thresholds given
    """

    if (sobel_kernel % 2) != 1:
        raise Exception('sobel_kernel must be an odd number')

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    sobel_mag = np.sqrt(np.square(sobelx) + np.square(sobely))

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * sobel_mag / np.max(sobel_mag))

    # 5) Create a binary mask where mag thresholds are met
    binary_output = binary_mask(C=scaled_sobel, thresh=mag_thresh)

    # 6) Return this mask as your binary_output image
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi / 2.)):
    """
    Apply Sobel x and y, then computes the direction of the gradient
    and apply a threshold.
    :param img: image to operate on 
    :param sobel_kernel: sobel_kernel size
    :param thresh: pair tuple of angles - in radians -  thresholds (low, high) where
    each value is in [0, pi/2] range. We're pretty much looking for vertical/horizontal
    directions, regarless of which quadrant the angle falls on 
    :return: 
    Binary image (mask) indicating the gradients with direction which meets the
    direction thresholds
    """

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    angles = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = binary_mask(C=angles, thresh=thresh)

    # 6) Return this mask as your binary_output image
    return binary_output

