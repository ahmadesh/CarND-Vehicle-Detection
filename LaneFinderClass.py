
# coding: utf-8

# ## Calibrating Camera

# In[1]:


import numpy as np
import cv2 
import pickle


class LaneFinder():
    def __init__(self):
        
        self.left_line = Line()
        self.right_line = Line()
        
        with open('camera.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
            self.mtx, self.dist, self.M, self.Minv = pickle.load(f)
        
# In[2]:


# Undistortion function
    def undist(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


# In[3]:

    def warping(self,img):
        img_size = (img.shape[1],img.shape[0])
        return cv2.warpPerspective(img, self.M, img_size, flags = cv2.INTER_LINEAR)


# In[4]:

    def pipeline(self, original, s_thresh=(120, 255), sx_thresh=(100, 255),l_thresh=(40,255)):
        img = np.copy(original)
        img = self.undist(img)
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
        # Threshold xy gradient
        sxybinary = np.zeros_like(scaled_sobel)
        sxybinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
        # Threshold lightness
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
        # Combining the thresholds
        combined = np.zeros_like(sxbinary)
        combined[(l_binary == 1) & (s_binary == 1) | (sxbinary == 1)] = 1
        return combined


# In[5]:


    def slidingWindow(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
    
        # lanes should be detected within a margin from left and right
        hist_margin = 100
        leftx_base = np.argmax(histogram[hist_margin:midpoint]) + hist_margin
        rightx_base = np.argmax(histogram[midpoint: histogram.shape[0] - hist_margin]) + midpoint 


        # Choose the number of sliding windows
        nwindows = 10
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 80
        # Set minimum number of pixels found to recenter window
        minpix = 40
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
    
        left_rectangles = []
        right_rectangles = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            left_rectangles.append((win_xleft_low,win_y_low, win_xleft_high, win_y_high)) 
            right_rectangles.append((win_xright_low,win_y_low, win_xright_high,win_y_high)) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
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

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    
        plotRects = (left_rectangles, right_rectangles)
        return left_fit, right_fit, left_lane_inds, right_lane_inds, plotRects


# In[6]:

    def laneWithwindow(self, binary_warped, left_fit, right_fit):  
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 50
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
        right_fit_new = np.polyfit(righty, rightx, 2)
    
        return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds


# In[7]:


    def curvature(self, binary_warped, left_fit, right_fit, left_lane_inds, right_lane_inds):
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
    
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        h = binary_warped.shape[0]
        ploty = np.linspace(0, h-1, h)
        ploty = np.linspace(0, 719, num=720)
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
        # Calculating the offset
        pos = binary_warped.shape[1]/2
        left0 = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        right0 = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        center = (left0 + right0) / 2.0
        offset = (pos - center) * xm_per_pix
    
        return left_curverad, right_curverad, offset


# In[8]:

    def addDrawing(self, original, warped, left_fit, right_fit):
        # Create an image to draw the lines on
        undist = np.copy(original)
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
        # Recast the x and y points into usable format for cv2.fillPoly()
        h = warped.shape[0]
        ploty = np.linspace(0, h-1, h)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (original.shape[1], original.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result        


# In[9]:
    
    def addText(self, original, rad, offset):
        undist = np.copy(original)
        h = undist.shape[0]
        font = cv2.FONT_HERSHEY_DUPLEX
        text = 'Radius of Curvature = ' + '{:04.2f}'.format(rad) + '(m)'
        cv2.putText(undist, text, (40,60), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
        LR = ''
        if offset > 0:
            LR = 'right'
        elif offset < 0:
            LR = 'left'
        text = 'Vehecle is ' + '{:04.2f}'.format(abs(offset)) + 'm ' + LR + ' of center'
        cv2.putText(undist, text, (40,120), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
        return undist


# In[15]:


    def addLineWarped(self, warped, left_fit, right_fit): 
        imcopy = np.copy(warped)
        # Recast the x and y points into usable format for cv2.fillPoly()
        h = warped.shape[0]
        ploty = np.linspace(0, h-1, h)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

        # Draw the lane onto the warped blank image
        cv2.polylines(imcopy, np.int_(pts_left), 0, (255, 0, 0), thickness=5)
        cv2.polylines(imcopy, np.int_(pts_right), 0, (255, 0, 0), thickness=5)

        return imcopy


# In[51]:

    def imageProcess(self, original):
        image = np.copy(original)
        binary = self.pipeline(image)
        warped = self.warping(image)
        binary_warped = self.warping(binary)
    
        if self.left_line.detected and self.right_line.detected:
            left_fit, right_fit, left_lane_inds, right_lane_inds = self.laneWithwindow(binary_warped, self.left_line.best_fit, self.right_line.best_fit)
        else:
            left_fit, right_fit, left_lane_inds, right_lane_inds, plotRects = self.slidingWindow(binary_warped)
    
        if left_fit is not None and right_fit is not None:
            # calculate x-intercept. The detected deference between left and right needs to be within a margin.
            h = image.shape[0]
            l_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
            r_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
            x_int_diff = abs(r_fit_x_int-l_fit_x_int)
            if abs(350 - x_int_diff) > 100:
                left_fit = None
                right_fit = None 
    
        self.left_line.update(left_fit, left_lane_inds)
        self.right_line.update(right_fit, right_lane_inds)
 
        if self.left_line.best_fit is not None and self.right_line.best_fit is not None:
            reimage = self.addDrawing(image, binary_warped, self.left_line.best_fit, self.right_line.best_fit)
            left_curverad, right_curverad, offset = self.curvature(binary_warped, self.left_line.best_fit, self.right_line.best_fit, left_lane_inds, right_lane_inds)
            outimage = self.addText(reimage, (left_curverad+right_curverad)/2., offset)
            warped_line = self.addLineWarped(warped, self.left_line.best_fit, self.right_line.best_fit)
        else:    
            outimage = image
        
        return outimage

# In[10]:

class Line():
    nn=0
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
    def update(self, fit, inds):
        if fit is not None:            
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit-self.best_fit)
            if (self.diffs[0] > 0.001 or                 self.diffs[1] > 0.8 or                 self.diffs[2] > 70.) and                 len(self.current_fit) > 0:
                self.detected = False
            else:
                self.detected = True
                self.current_fit.append(fit)
                if len(self.current_fit) > 5:
                    # replaceing fits with new ones
                    self.current_fit = self.current_fit[len(self.current_fit) - 5:]
                self.best_fit = np.average(self.current_fit, axis=0)
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)





# In[ ]:




