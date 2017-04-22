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
import LineFindingTools
from scipy import stats
import math

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, nbuff=3, ord=2):

        # #average x values of the fitted line over the last n iterations
        # self.bestx = None

        # #polynomial coefficients averaged over the last n iterations
        # self.best_fit = None


        # #distance in meters of vehicle center from the line
        # self.line_base_pos = None

        # Buffer size
        self.buff_size = nbuff

        # Fitted polynomial order
        self.ord = ord

        # Reset allocates buffers
        self.reset()

    def onFit(self, fit, yrange):
        """
            Called on new obtain fit
        :param fit: proposed fit to add to buffer 
        :return: 
        True if fit has been added
        """

        # Add new fit to buffer
        if(self._ok_to_add_fit(fit, yrange)):
            self._add_buffered_fit(fit, yrange)
            self.detected = True
            return True
        else:
            self.add_fails += 1
            return False



    def reset(self):
        """
        Reset values pertaining to a valid lane representation
        :return: 
        Nothing
        """

        # was the line detected in the last iteration?
        self.detected = False

        # x values for detected line pixels
        self.allx = []
        # y values for detected line pixels
        self.ally = []
        for i in range(0, self.buff_size):
            self.allx.append([])
            self.ally.append([])

        # x values of the last n fits of the line
        self.recent_xfitted = []

        # difference in fit coefficients between last and new fits
        self.diffs = np.zeros(shape=(1, self.ord), dtype=np.float64)

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # Line coefficient buffer
        self.coeff_buff = np.zeros(shape=(self.buff_size, self.ord + 1), dtype=np.float64)

        # Line coeff init
        self.coeff_buff_init = np.zeros(shape=(self.buff_size, 1), dtype=np.bool)

        # Count the number of failures to add
        self.add_fails = 0

    def _ok_to_add_fit(self, fit, yrange):
        """
        Make sure that new fits being added make sense 
        - No large jumps  between last and current
        
        # TBD: Check what the average expected norm is normally
        
        :param fit: fit to be added
        :return: 
        OK - True if OK, False otherwise
        """


        # If we already have a fit in the buffer, compare with last one
        # otherwise, this is the first fit we've obtained
        if self.coeff_buff_init[-1,0]:
            # How does this fit's values compare to the values we already have
            xnp = [x for x in self.allx if x != []]
            xnp = np.array(xnp).flatten()
            xrng = (np.min(xnp), np.max(xnp))
            xmid = np.median(xnp)

            fxvals = LineFindingTools.poly_eval(fit, yrange, ord=self.ord)
            fxmed = np.median(fxvals)
            # Tolerate a couple of orders of existing range x-diffs. Otherwise, not OK
            # This fit may be faulty

            # is this fit's median point within the previous median +/- 1.5 their range ?
            v = (fxmed > (xmid - 1.5*xrng[0])) and (fxmed < (xmid + 1.5*xrng[1]))

            return v

        else:
            # First entry
            return True

    def get_buffered_fit(self):
        """
        Return the fit over the buffered vals
        Shortcut: the average of the coefficients over the buffer
        :return: 
        Fit over buffered values (should we send just the average of coeff instead ??)
        """
        # xvals = []
        # yvals = []
        # for i in range(0, self.buff_size):
        #     if self.coeff_buff_init[i]:
        #         xvals.append(self.allx[i])
        #         yvals.append(self.ally[i])
        #
        # # Y is the independent variable
        # buff_fit = np.polyfit(np.array(yvals).flatten(), np.array(xvals).flatten(), self.ord)

        buff_fit = np.mean(self.coeff_buff[self.coeff_buff_init.flatten(), :], axis=0)

        # n = np.sum(self.coeff_buff_init)
        # buff_fit = np.sum(self.coeff_buff, axis=0)/np.float64(n)
        return buff_fit

    def _add_buffered_fit(self, fit, yrange):
        """
            Add new fit to buffer
        :param fit: 
        :return: 
        """

        self.recent_xfitted = LineFindingTools.poly_eval(fit, yrange, ord=self.ord)
        # Accumulate new fit and values
        # Shift existing entries up
        for i in range(0, self.buff_size - 1):
            self.coeff_buff[i, :] = self.coeff_buff[i + 1, :]
            self.coeff_buff_init[i] = self.coeff_buff_init[i + 1]
            self.allx[i] = self.allx[i+1]
            self.ally[i] = self.ally[i+1]

        # Insert new fit and values at the bottom
        self.coeff_buff[-1, :] = fit
        self.allx[-1] = self.recent_xfitted
        self.ally[-1] = yrange
        self.coeff_buff_init[-1, :] = True



def LineFactory(ord = 2, nbuff = 3):
    """
    Create the lines needed for this project
    Left/Left Scaled .. & so on
    :param ord: Order of fit
    :param nbuff: buffer size
    :return: 
    Dictionary of lines with same parameters
    """
    lines = {'left': Line(nbuff=nbuff, ord=ord),
             'left_scaled':Line(nbuff=nbuff, ord=ord),
             'right': Line(nbuff=nbuff, ord=ord),
             'right_scaled': Line(nbuff=nbuff, ord=ord),
             'center': Line(nbuff=nbuff, ord=ord),
             'center_scaled':Line(nbuff=nbuff, ord=ord)}

    return lines


