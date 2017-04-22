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
from scipy.odr import Model, ODR, Data

from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def orthoregress(x, y, ord):
    """Perform an Orthogonal Distance Regression on the given data,
    using the same interface as the standard scipy.stats.linregress function.
    Arguments:
    x: x data (input)
    y: y data (response)
    Returns:
        
    a[], where: a[0]*x^ord + a[1]*x^(1) + .... + a[ord]
        
    Uses standard ordinary least squares to estimate the starting parameters
    then uses the scipy.odr interface to the ODRPACK Fortran code to do the
    orthogonal distance calculations.

    Source: http://blog.rtwilson.com/orthogonal-distance-regression-in-python/
    """
    poly_fit = np.polyfit(x, y, ord)
    # mod = Model(f)
    mod = Model(lambda pp, xx:poly_eval(pp, xx, ord))
    dat = Data(x, y)
    od = ODR(data = dat, model = mod, beta0=poly_fit[0:ord + 1], maxit=10)
    out = od.run()
    return out.beta

def poly_eval(p,x,ord):
    """
    Polynomial evaluation
    """
    val = np.zeros(shape=x.shape, dtype=p.dtype)

    for i in range(0, ord + 1):
        val += p[i]*(np.power(x, ord - i))

    # return (p[0] * x * x) + p[1] * x + p[2]
    return val

def RANSAC_Fit(x, y, ord):
    """
        Fit a n-th order polynomial to a 2D set of datapoints. X is the input, Y is the dependent 
        variable.
    :param x: input variable
    :param y: dependent variable
    :param ord: linear order of polynomial
    :return: 
    polynomial coefficients 'a' as: a[0]*x^ord + a[1]*x^(ord - 1) + .... + a[ord]
    """
    min_x = np.int64(np.floor(np.min(x)))
    max_x = np.int64(np.ceil(np.max(x)))
    min_y = np.int64(np.floor(np.min(y)))
    max_y = np.int64(np.ceil(np.max(y)))

    # Fit line using all data
    # model = linear_model.LinearRegression()

    model = make_pipeline(PolynomialFeatures(ord), Ridge(alpha=5.0, fit_intercept = True))
    # model.fit(x, y)

    # model = Pipeline([('poly', PolynomialFeatures(degree=ord)), ...
    #                   ('linear', LinearRegression(fit_intercept=True))])

    # model = make_pipeline(PolynomialFeatures(ord), LinearRegression(fit_intercept=True))

    # Robustly fit linear model with RANSAC algorithm
    model_ransac = linear_model.RANSACRegressor(base_estimator=model)
    model_ransac.fit(np.expand_dims(x, axis=1), np.expand_dims(y, axis=1))

    # inlier_mask = model_ransac.inlier_mask_
    # outlier_mask = np.logical_not(inlier_mask)

    # Compute the points generated within the x-range (min/max)
    x_range = np.array(range(min_x, max_x + 1))
    x_range = x_range.reshape(len(x_range), -1)

    # Predict data of estimated models
    y_range = np.array(np.around(model_ransac.predict(x_range), decimals=0), dtype=np.int64)

    # Fit an 'order' polynomial
    return orthoregress(x, y, ord)

def lanes_ok(LRfits, var):
    """
     !!! Expects scaled fits !!!
    Check if the generated lanes are OK.
    Three conditions are checked:
    1 - Curvatures are similar
    2 - Parallel
    3 - Average distance between the lines is within expectations
    
    :param fits - tuple pairs of left/right fits scaled to world measurements 
    :param var - independent variable range to check (the y-variable)
    :return: 
    OK - boolean if detected lanes make sense
    reason - reason for failure. Empty list if OK == True
    """
    OK = True
    reason = []

    left_fit = LRfits[0]
    right_fit = LRfits[1]

    assert len(left_fit) == len(right_fit)

    # fit is a polynomial as: a0 * x^(ord) + a1 * x^(ord - 1) + ... + aord
    ord = len(left_fit) - 1

    leftxrange = poly_eval(left_fit, var, ord=ord)
    leftcurvature = curvature(left_fit, var)
    rightxrange = poly_eval(right_fit, var, ord=ord)
    rightcurvature = curvature(right_fit, var)

    # Are lane curvatures similar ?
    # If radius of curvature is very large, lines are fairly straight, hence
    # they can be considered parallel (in the straight sense). If, however, the radius
    # of curvature is short, that means that there is significant turn, in which case
    # the range of curvature should be considered

    # if (np.mean(leftcurvature) > 10000.) and (np.mean(rightcurvature) > 10000.):
    #     # Very large radius, low curvature, both can be considered straight and parallel
    #     pass
    # else:
    #     # How large is the difference between curvatures
    #     dc = np.abs(leftcurvature - rightcurvature)
    #     cdmean = np.mean(dc)
    #     # If difference is in order of magnitude
    #     if cdmean > 10000.:
    #         OK = False
    #         reason.append('curvature')

    # Distances between lines representing lanes
    dh = np.abs(leftxrange - rightxrange)

    # Are the lanes parallel ?
    # Parallel check: is the distance between horizontal positions about the same
    # along the vertical positions (distance from the vehicle)
    min_dh = np.min(dh)
    max_dh = np.max(dh)
    # Largest deviation is greater than 1/10 th of the
    # lane width of about 5m
    # TBD !!!
    # if max_dh - min_dh > 0.5:
    #     OK = False
    #     reason.append('parallel')


    # Is the avg distance between the lanes within expectations ?
    # Expected minimum lane width in the US is 3.7m
    # Interstates have
    # Check that avg. lanes width falls within a certain window
    avgdist = np.mean(dh)
    if (avgdist < 3.5) or (avgdist > 5.2):
        OK = False
        reason.append('lane_width')

    return OK, reason

def radius_of_curvature(a, val):
    """
    Find the radius of curvature of a line, valued at 'val', of a line represented
    by as second order polynomial. The formula used:
    
    Line: y = f(a, val) = a[0] * val ^ 2 + a[1] * val + a[2]
    
    Rad of curvature: R(val) = {(1 + (df(a, val)/d_val) ^ 2 ) ^ (3/2)} / {abs( df2(a, val)/d2_val )}
    
    which results into:
    
    R = {(1 + (2*a[0]*val + a[1]) ^ 2) ^ (3/2)}  / abs(2 * a[0])
    
    :param a - 2nd order polynomial coefficients of form a[0] *x^2 + a[1] * x + a[2] 
    :param val - value to evaluate the radius of curvature at
    :return: 
    R - radius of curvature
    """

    m0 = (2*a[0]*val+a[1]) ** 2
    m1 = np.power(1 + m0, 1.5)
    R = m1/np.abs(2 * a[0])

    return R

def curvature(a, paramvals):
    """
    Generate a sequence of radii of curvature for a n-th order polynomial 
    defined line, along its dependent parameter
    :param a: polynomial line parameters, defined as 
    a[0] * paramval^n + a[1] * paramval^(n-1) + ... + a[n]
    :param paramvals: np array of parametervalues
    :return: 
    array of radii of curvatures along the parameter values
    """
    r = np.zeros_like(paramvals)
    nsamps = np.max(np.size(paramvals))
    for i in range(0, int(nsamps)):
        r[i] = radius_of_curvature(a, paramvals[i])

    return r

def genfit(nonzero, inds, ord):
    """
    Generate fits for the non-zero pixels at desired indeces
    :param nonzero: pair (tuple) of same length arrays of non-zero pixel locations 
                    (independent_var[], dependent_var[])
    :param inds: indeces of entries we want to generate fits for 
    :param ord: order of polynomial fit
    :return: 
    polynomial fit 'a' where: a[0] * independentvar ^ ord + ... + a[ord] 
    """

    independentvar = nonzero[0][inds]
    dependentvar = nonzero[1][inds]

    # Fit a second order polynomial to each
    # *
    # fit = np.polyfit(independentvar, dependentvar, ord)
    # *
    fit = orthoregress(independentvar, dependentvar, ord)
    # *
    # fit = RANSAC_Fit(independentvar, dependentvar, ord)

    return fit

def center_offset(center_fit, vehctr, indepvarmax):
    """
    Compute the center offset from the center of the lane.
    Center of lane computed as the polynomial of center_fit @ ymax
    where ymax corresponds to the current position of the vehicle
    along the visible lanes. This is the bottom of the camera image.

    To the left the offset is negative, to the right, the offset is positive.
    The returned scale depends on the input values.
    !! Note !!
    The function will compute blindly the values. Client code must
    make sure that the fit and the scales of the values are correct
    :param center_fit: center lane fit
    :param vehctr: vehicle center (in pixel of vehicle coordinates)
    :param indepvarmax: bottom of image (in pixel or scaled coordinates)
    :return: 
    offset measured in input values scale
    """

    ord = len(center_fit) - 1
    lane_ctr = poly_eval(center_fit, indepvarmax, ord)
    # If vehicle center is to the left of lane center, the offset value
    # will be negative, otherwise positive (or zero if values match)
    return vehctr - lane_ctr

def center_fit(fits, independvarrange, ord):
    """
    Find the center fit between L/R fits. The center fit will be calculated
    by fitting a a polynomial to the indepenent variables as (Left + Right) / 2
    then the calculated mid-points (dependent vars) along with the independent
    variable will be used to obtain a new fit
    :param fits: tuple pair of Left & Right fits
    :param independvarrange: range of indepenent variables (numpy array)
    :return: 
    middle_fit - mid-fit of the two input fits
    """

    left_fit = fits[0]
    right_fit = fits[1]
    left_poly_vals = poly_eval(left_fit, independvarrange, ord)
    right_poly_vals = poly_eval(right_fit, independvarrange, ord)
    mid_dependent_vars = (right_poly_vals + left_poly_vals)/2.0

    # Simple
    middle_fit = np.polyfit(independvarrange, mid_dependent_vars, ord)

    # middle_fit = RANSAC_Fit(independvarrange, mid_dependent_vars, ord)

    return middle_fit

def compute_fits(imgshape, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, xscale, yscale, ord):
    """
    Compute the left, middle, and right fits of lanes
    :param imgshape: 
    :param nonzerox: 
    :param nonzeroy: 
    :param left_lane_inds: 
    :param right_lane_inds: 
    :param xscale: 
    :param yscale: 
    :param ord: 
    :return: 
    """

    # y is our independent variable here
    left_fit = genfit((nonzeroy, nonzerox), left_lane_inds, ord=ord)
    right_fit = genfit((nonzeroy, nonzerox), right_lane_inds, ord=ord)
    left_fit_scaled = genfit((nonzeroy * yscale, nonzerox * xscale), left_lane_inds, ord=ord)
    right_fit_scaled = genfit((nonzeroy * yscale, nonzerox * xscale), right_lane_inds, ord=ord)

    # Mid-lane fit
    yrange = np.linspace(0, imgshape[0] - 1, imgshape[0])
    yrange_scaled = yrange * yscale
    middle_fit = center_fit((left_fit, right_fit), independvarrange=yrange, ord=ord)
    middle_fit_scaled = center_fit((left_fit_scaled, right_fit_scaled), independvarrange=yrange_scaled, ord=ord)

    return left_fit, left_fit_scaled, right_fit, right_fit_scaled, middle_fit, middle_fit_scaled

def sliding_window_lanes(img, nwindows=9, xscale=1, yscale=1, ord = 2):
    """
        Use sliding windows (vertically) to find lane pixels.
         Fit a second order polynomial nto the found pixels. 
         Scale polynomials to input x/y scaling factors (e.g. to world 
         measures)
    :param img: binary input image
    :param nwindows: windows to scan upwards along the line points
    :param xscale / yscale : pixel to other measure (for example world)
    :return: 
    out_img -  image with rectangles drawn on it (for reference and visualization)
    left_fit, left_fit_scaled, - pixel/scaled fits for left lane
    right_fit, right_fit_scaled - pixel/scale fits for right lane
    nonzerox, nonzeroy - paired arrays for locations of non-zero pixels
    left_lane_inds, right_lane_inds - indeces of pixels belonging to the lanes in the "nonzerox, nonzeroy" pairs
    """

    # Order of polynomial fit
    out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = \
        sliding_window_lane_pixels(img, nwindows)

    left_fit, left_fit_scaled, right_fit, right_fit_scaled, middle_fit, middle_fit_scaled = \
    compute_fits(img.shape, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, xscale, yscale, ord)

    # Compute offset from center
    yrange = np.linspace(0, img.shape[0] - 1, img.shape[0])
    yrange_scaled = yrange * yscale
    # ctr_offset = center_offset(middle_fit_scaled, np.float64(img.shape[1]) * xscale / 2., np.max(yrange_scaled))

    # Do the scaled measurements make sense ?
    OK, reason = lanes_ok((left_fit_scaled, right_fit_scaled), var=yrange_scaled)

    return out_img, left_fit, left_fit_scaled, right_fit, right_fit_scaled, \
           middle_fit, middle_fit_scaled, \
           OK, reason

def sliding_window_lane_pixels(img, nwindows):
    """
        Find the lanes in an image. This function assumes that the center of the lanes is
         around the center of the image. A moving vertically sliding window is implemented
         to search the lane pixels. 

    :param img: binary image to search lanes in.
    :param nwindows: number of windows (vertical count, i.e. along the y/height axis)
    :return: 
    out_img - image with the windows painted on it (for visualization purposes)
    nonzerox, nonzeroy - the pair lane-pixel coordinates
    left_lane_inds, right_lane_inds - left / right lane indeces of non-zero pixels
    """

    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[np.int(img.shape[0] / 2):, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one and search for pixels that belong to
    # the lanes
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) &
                          (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) &
                           (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Paint non-zero pixels, belonging to the line-windows with colors
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds

def lane_pixels_indeces_around_fit(nonzero, fits, ord, margin = 100):
    """
    Find pixels of lanes around a L/R fit
    :param nonzero: nonzero pixels
    :param fits: tuple or fits (left_fit, right_fit) of same order
    :param ord: order of fit being applied
    :return: 
    """

    left_fit = fits[0]
    right_fit = fits[1]
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # left_poly_vals = np.zeros(shape=nonzeroy.shape, dtype=np.float64)
    # right_poly_vals = np.zeros(shape=nonzeroy.shape, dtype=np.float64)
    #
    # # Apply the polynomial p(independent_var) to obtain the dependent variable values
    # for i in range(0, ord + 1):
    #     left_poly_vals += left_fit[i]*(np.power(nonzeroy, ord-i))
    #     right_poly_vals += right_fit[i]*(np.power(nonzeroy, ord-i))

    left_poly_vals = poly_eval(left_fit, nonzeroy, ord)
    right_poly_vals = poly_eval(right_fit, nonzeroy, ord)

    # Find the indeces of non-zero pixels around the polynomial - within the margin
    left_lane_inds = (nonzerox > (left_poly_vals - margin)) & (nonzerox < (left_poly_vals + margin))
    right_lane_inds = (nonzerox > (right_poly_vals - margin)) & (nonzerox < (right_poly_vals + margin))

    return left_lane_inds, right_lane_inds

def lanes_viz(img, fits, nonzerox, nonzeroy, lane_inds, margin, ctroffset, ord):

    """
        Visualize lanes onto image. Utility function
    :param img: 
    :param fits: 
    :param nonzerox: 
    :param nonzeroy: 
    :param lane_inds: 
    :param margin: 
    :return: 
    Image with visualizations on it
    """

    left_fit, right_fit = fits[0], fits[1]
    left_lane_inds, right_lane_inds = lane_inds[0], lane_inds[1]

    ######## visualizations ######
    out_img = np.dstack((img, img, img)) * 255
    window_img = np.zeros_like(out_img)

    # Generate x and y values for plotting
    ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
    left_fitx = poly_eval(left_fit, ploty, ord)
    right_fitx = poly_eval(right_fit, ploty, ord)

    # Paint non-zero pixels, belonging to the line-windows with colors
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return out_img

def print_lane_info(img, rad_curv, veh_offset):
    """
    Embed in textual form lane info in image
    :param img: image to embed info into
    :param rad_curv: radius_of_curvature(m)
    :param veh_offset: vehcile offset (m)
    :return: none
    """

    # Print the offset from center lane
    font = cv2.FONT_HERSHEY_SIMPLEX
    # xpos = np.int32(img.shape[1] / 2. - img.shape[1] / 3.)
    xpos = 50
    cv2.putText(img, 'Curv: {0:.2f}km, Ctr. Offset: {1:.2f}m'.format(np.float64(rad_curv/1000.),
                                                                np.float64(veh_offset)),
                (xpos, 100), font, 2, (255, 255, 255), 3)

def lanes(img, initfits, xscale=1, yscale=1, ord=2, viz=True):
    """
    Find pixels in the binary image starting the search from initial fits
    :param img: binary image
    :param initfits: tuple of L/R previous fits to be used as initial starting points (pixel domain) 
    :param ord: order of polynomial used for fitting lanes
    :return: 
    out_img -  image with rectangles drawn on it (for reference and visualization)
    left_fit, left_fit_scaled, - pixel/scaled fits for left lane
    right_fit, right_fit_scaled - pixel/scale fits for right lane
    nonzerox, nonzeroy - paired arrays for locations of non-zero pixels
    left_lane_inds, right_lane_inds - indeces of pixels belonging to the lanes in the "nonzerox, nonzeroy" pairs
    """
    margin = 100
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds, right_lane_inds = lane_pixels_indeces_around_fit(nonzero, initfits, ord=ord, margin=margin)
    left_fit, left_fit_scaled, right_fit, right_fit_scaled, middle_fit, middle_fit_scaled = \
        compute_fits(img.shape, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, xscale, yscale, ord)

    yrange = np.linspace(0, img.shape[0] - 1, img.shape[0])
    yrange_scaled = yrange * yscale

    # Do the scaled measurements make sense ?
    OK, reason = lanes_ok((left_fit_scaled, right_fit_scaled), var=yrange_scaled)


    ######## visualizations ######
    if viz:
        ctr_offset = center_offset(middle_fit_scaled, np.float64(img.shape[1]) * xscale / 2., np.max(yrange_scaled))
        out_img = lanes_viz(img, fits=(left_fit, right_fit), lane_inds=(left_lane_inds, right_lane_inds),
                             nonzerox=nonzerox, nonzeroy=nonzeroy, margin=margin, ctroffset=ctr_offset, ord=ord)
    else:
        out_img = np.copy(img)

    return out_img, left_fit, left_fit_scaled, \
           right_fit, right_fit_scaled, \
           middle_fit, middle_fit_scaled, OK, reason







