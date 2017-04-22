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
import glob
import math
from ImgTools import draw_lines
import matplotlib.pyplot as plt


def cal_undistort(img, objpoints, imgpoints):
    '''
    :param img: distoreted image
    :param objpoints: 
    :param imgpoints: 
    :return: 
        undistimg - undistorted image
    '''

    img_size = (img.shape[1], img.shape[0])
    # Use cv2.calibrateCamera() and cv2.undistort()
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size ,None ,None)
    undistimg=  cv2. undistort(img, mtx, dist, None, mtx)
    return undistimg, mtx, dist, rvecs, tvecs

def chessboard_corners(filename, nx, ny):
    '''
    Generate the corners of the chessboards. Epxect images 
    to be RGB

    :param filename: file (or wildcard) image file names
    :param nx: number of inside corners in x
    :param ny: number of inside corners in y
    :return: 
        objpoints : 3d points in real world space (from many checkboard images)
        cornerpoints : 2d corner points in image plane. (the inner corners on the image.)
    '''

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(nx,ny,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    # These lists maintain world coords & inner corner locations for each image.
    # Each entry corresponds to an image:
    # objpoints[i] - world coords for image i
    # imgpoints[i] - corner locations ON image i
    objpoints = []  # 3d points in real world space (from many checkboard images) - discrete coordinates system,
    cornerpoints = []  # 2d corner points in image plane. (the inner corners on the image.)

    # Make a list of calibration images
    images = glob.glob(filename)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            cornerpoints.append(corners)

    return objpoints, cornerpoints

def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps

    # 1) Undistort using mtx and dist
    undistimg = cv2.undistort(img, mtx, dist, None, mtx)

    # 2) Convert to grayscale
    gray = cv2.cvtColor(undistimg, cv2.COLOR_BGR2GRAY)

    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # 4) If corners found:
    if ret:
        # a) draw corners
        cv2.drawChessboardCorners(gray, (nx, ny), corners, ret)

        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
        # Note: you could pick any four of the detected corners
        # as long as those four corners define a rectangle
        # One especially smart way to do this would be to use four well-chosen
        # corners that were automatically detected during the undistortion steps
        # We recommend using the automatic detection of corners in your code
        # src = np.float32([corners[0], corners[nx-1], corners[(ny-1)*(nx)], corners[-1]])
        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])

        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100  # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                          [img_size[0] - offset, img_size[1] - offset],
                          [offset, img_size[1] - offset]])

        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)

        # e) use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(undistimg, M, img_size)
    else:
        M = None
        warped = np.copy(img)

    return warped, M

def trapezius_roi(imgshape, img, narrow = False):
    x_idc = 0
    y_idc = 1

    if narrow:
        apex_y_offset = 100
        apex_x_l_offset = 64
        apex_x_r_offset = 66
    else:
        apex_y_offset = 100
        apex_x_l_offset = 84
        apex_x_r_offset = 86


    ctr_x = int(math.floor(imgshape[1] / 2.))
    ctr_y = int(math.floor(imgshape[0] / 2.))
    max_x = int(imgshape[1])
    max_y = imgshape[0]


    # ROI Definition
    left_bottom = (200, max_y)
    right_bottom = (max_x - 160, max_y)
    apex = (ctr_x + 2, ctr_y + apex_y_offset)

    p1 = (apex[x_idc] - apex_x_l_offset, apex[y_idc])
    p2 = (apex[x_idc] + apex_x_r_offset, apex[y_idc])
    p3 = (right_bottom[x_idc], right_bottom[y_idc])
    p4 = (left_bottom[x_idc], left_bottom[y_idc])
    verts = np.array([p1, p2, p3, p4], dtype=np.float32)

    # ************ Debugging: Draw ROI bounding lines on the raw image *********
    l1 = [left_bottom[x_idc], left_bottom[y_idc], apex[x_idc] - apex_x_l_offset, apex[y_idc]]
    l2 = [apex[x_idc] - apex_x_l_offset, apex[y_idc], apex[x_idc] + apex_x_r_offset, apex[y_idc]]
    l3 = [apex[x_idc] + apex_x_r_offset, apex[y_idc], right_bottom[x_idc], right_bottom[y_idc]]
    l4 = [right_bottom[x_idc], right_bottom[y_idc], left_bottom[x_idc], left_bottom[y_idc]]
    L = [l1, l2, l3, l4]
    roi_lines_image = np.copy(img)
    draw_lines(img=roi_lines_image, lines=L, color=[255, 0, 0], thickness=2)
    # plt.imshow(roi_lines_image)
    # *************************************************************

    return verts, roi_lines_image

def warp_img(img, objpoints, imgpoints, narrow = False):
    """
        Given the calibration parameters, warp image.
    :param img: image being worked with
    :param objpoints: points (inner corners in the world) obtained from chessboard calibration
    :param imgpoints: points (inner corners in the image) obtained from chessboard calibrtaion
    :return: 
    warped image
    """

    # 1) Undistort (using calibration params)
    undistimg, _, _, _, _ = cal_undistort(img=img, objpoints=objpoints, imgpoints=imgpoints)

    # Trapezius ROI becomes square after warping
    src, roi_lines_image = trapezius_roi(img.shape, undistimg, narrow)

    # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    # These are the reference where the ROI will be transformed to
    # Choose offset from image corners to plot detected corners
    # This should be chosen to present the result at the proper aspect ratio
    # My choice of 100 pixels is not exact, but close enough for our purpose here
    offset = 0  # offset for dst points

    # Grab the image shape
    # image size: <x, y> = <width, height>
    img_size = (undistimg.shape[1], undistimg.shape[0])

    # dst is ordered as <x1, y1>, <x2, y2>, .... where y - height and x - width of image
    dst = np.float32([[200, 0], [img_size[0] - 200, offset],
                      [img_size[0] - 200, img_size[1] - offset],
                      [200, img_size[1] - offset]])

    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Inverse transform
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(undistimg, M, img_size, flags=cv2.INTER_LINEAR)

    ### Debug
    # warped_lines_image = cv2.warpPerspective(roi_lines_image, M, img_size, flags=cv2.INTER_LINEAR)
    # plt.imshow(warped_lines_image)

    return warped, M, Minv, roi_lines_image

def unwarp_img(color_warp, Minv):
    """
    Unwarp image using the inverse of the transformation matrix
    used to warp the image.
    :param color_warp: 
    :param Minv: 
    :return: 
    """

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwarped = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0]), flags=cv2.INTER_LINEAR)

    return unwarped