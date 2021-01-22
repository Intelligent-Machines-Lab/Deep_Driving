from __future__ import absolute_import, division, print_function
import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

sys.path.append(ros_path)
import math
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, sqrt
import control
import time


def canny(image, plot):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    if plot == True:
        cv2.imshow('gray', gray)
        cv2.imshow('blur', blur)
        cv2.imshow('canny', canny)
        cv2.waitKey(0)
    return canny


def roi(image, plot):
    height = image.shape[0]
    polygons = np.array([
        [(0, 480), (60, 310), (580, 310), (640, 480)]
    ])  # to be actually defined
    mask = np.zeros_like(image)
    # cv2.polylines(mask,polygons, True, 255)
    cv2.fillPoly(mask, polygons, 255)
    # cv2.imshow('mask', mask)
    masked = cv2.bitwise_and(image, mask)
    if plot == True:
        cv2.imshow('mask', mask)
        cv2.imshow('masked', masked)
        cv2.waitKey(0)
    return masked


def speed(prediction):
    pwm_max = 10
    pwm_base = 0.5 * pwm_max
    if prediction > 0.5:
        left = (1.5 - prediction) * (1.5 - prediction) * pwm_base
        right = (1.5 - prediction) * pwm_base
    else:
        right = (-1 + 4 * prediction) * (0.5 + prediction) * pwm_base
        left = (1.5 - prediction) * pwm_base

    return (left, right)


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Or use RGB2GRAY if you read an image with mpimg


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def x_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = grayscale(img)
    # Take only Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    # Calculate the absolute value of the x derivative:
    abs_sobelx = np.absolute(sobelx)
    # Convert the absolute value image to 8-bit:
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # Create binary image using thresholding
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = grayscale(img)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = grayscale(img)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    dir_grad = np.arctan2(abs_sobely, abs_sobelx)

    binary_output = np.zeros_like(dir_grad)
    binary_output[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1

    return binary_output


def hsv_select(img, thresh_low, thresh_high):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    color_binary = np.zeros((img.shape[0], img.shape[1]))
    color_binary[(hsv[:, :, 0] >= thresh_low[0]) & (hsv[:, :, 0] <= thresh_high[0])
                 & (hsv[:, :, 1] >= thresh_low[1]) & (hsv[:, :, 1] <= thresh_high[1])
                 & (hsv[:, :, 2] >= thresh_low[2]) & (hsv[:, :, 2] <= thresh_high[2])] = 1
    return color_binary


def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    s = hls[:, :, 2]
    s_binary = np.zeros_like(s)
    s_binary[(s > thresh[0]) & (s <= thresh[1])] = 1
    return s_binary


def warp(img):
    img_size = (img.shape[1], img.shape[0])
    IMAGE_H = 480
    IMAGE_W = 640

   # src = np.float32([[0, 220], [IMAGE_W, 220], [180, 0], [420, 0]]) #[[0, 220], [IMAGE_W, 220], [180, 0], [420, 0]]
   # dst = np.float32([[170, 220], [480, 220], [180, 0], [420, 0]]) #[[170, 220], [475, 220], [180, 0], [420, 0]]
    src = np.float32([[0, 240], [320, 240], [0, 0], [320, 0]])  # [[0, 220], [IMAGE_W, 220], [180, 0], [420, 0]]
    dst = np.float32([[120, 240], [205, 240], [0, 0], [320, 0]])  # [[170, 220], [475, 220], [180, 0], [420, 0]]
    M = cv2.getPerspectiveTransform(src, dst)

    # inverse
    Minv = cv2.getPerspectiveTransform(dst, src)

    # create a warped image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    unpersp = cv2.warpPerspective(warped, Minv, img_size, flags=cv2.INTER_LINEAR)

    return warped, unpersp, Minv


def lane_detector(image, video_mode=False):
    # read image
    # if video_mode == False:
    #    image = cv2.imread(image)

    # Undistort image
    undist = image
    # print(undist.shape)

    # Define a kernel size and apply Gaussian smoothing
    apply_blur = True
    if apply_blur:
        kernel_size = 5
        undist = gaussian_blur(undist, kernel_size)

    # Define parameters for gradient thresholding
    sxbinary = x_thresh(undist, sobel_kernel=3, thresh=(90, 100)) #22,100 90,100
    mag_binary = mag_thresh(undist, sobel_kernel=3, thresh=(80, 100)) #40,100 80,100
    #dir_binary = dir_threshold(undist, sobel_kernel=15, thresh=(0.7, 1.3))

    # Define parameters for color thresholding
    #s_binary = hls_select(undist, thresh=(180, 255))
    s_binary = hsv_select(undist, (0,0,200), (179,17,255))
    #plt.imshow(s_binary)
    #plt.show()
    # You can combine various thresholding operations

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    #color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary1 = np.zeros_like(sxbinary)
    combined_binary1[(s_binary == 1) | (sxbinary == 1)] = 1
    # combined_binary1 = s_binary
    #combined_binary2 = np.zeros_like(sxbinary)
    #combined_binary2[(s_binary == 1) | (sxbinary == 1) | (mag_binary == 1)] = 1

    # Apply perspective transform
    # Define points
    warped_im, _, Minv = warp(combined_binary1)

    return undist, sxbinary, s_binary, combined_binary1, warped_im, Minv


def fit_lines(img, plot=True):
    binary_warped = img.copy()
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(0):, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    # Make this more robust
    #print('0')
    midpoint = np.int(histogram.shape[0] / 2)  # lanes aren't always centered in the image
    #print('1')
    leftx_base = np.argmax(histogram[0:midpoint])  # Left lane shouldn't be searched from zero
    #print("bases",leftx_base,histogram.argmax())
    #print('2')
    rightx_base = np.argmax(histogram[midpoint: midpoint + 500]) + midpoint
    #print('3')
    # Choose the number of sliding windows
    if rightx_base == midpoint:
        rightx_base = 400
    if leftx_base == rightx_base:
        leftx_base = 241
    if rightx_base - leftx_base < 90:
        #print("aqui",rightx_base - leftx_base)
        if (400 - rightx_base) > (rightx_base):
            rightx_base = 400
        if (leftx_base) > (400 - leftx_base):
            leftx_base = 241
    #print('4')
    nwindows = 6
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 20
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    left_windows = []
    right_windows = []
    i = 1
    #print('5')
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        #if len(good_left_inds)>minpix:
        #    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        #if len(good_right_inds)>minpix:
        #    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 0, 0), 2)
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            left_windows.append(i)
        else:
            left_windows.append(0)
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            right_windows.append(i)
        else:
            right_windows.append(0)
        i = i+1
    #print('6')
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
   # print('ind',len(left_lane_inds),len(right_lane_inds))
    #print('7')
    if (np.count_nonzero(left_windows)>5):

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        left_fit_null = 0
        #print("here1")
    else:
        left_fit = 0
        left_fit_null = 1
        left_fitx = 0 * ploty ** 2 + 0 * ploty + 0
        #print("here2")
    #print('8')
    if (np.count_nonzero(right_windows)>5):

        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        #right_fitx_ex = right_fit[0]*(ploty)**2 + right_fit[1]*(ploty)+right_fit[2] -320
        right_fit_null = 0
        #print("here3")
    else:
        right_fit = 0
        right_fit_null = 1
        right_fitx = 0 * ploty ** 2 + 0 * ploty + 320
        #print("here4")
    #print('9')
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #print('10')
    if len(lefty)> 0:
        aux_l = np.max(lefty)
    else:
        aux_l = 0

    if len(righty)>0:
        aux_r = np.max(righty)
    else:
        aux_r = 0


   # print(left_windows,right_windows)
    if plot == True:
        #fig = plt.figure()
        #print(right_fitx[0], right_fitx[1], right_fitx[2])
        plt.imshow(binary_warped)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 640)
        plt.ylim(480, 0)
        plt.draw()
        plt.show(block = False)
        plt.pause(0.0005)
        plt.clf()
    return left_fit, left_fit_null, right_fit, right_fit_null, aux_l, aux_r, out_img

# Calculate Curvature
def curvature(left_fit, left_fit_null, right_fit, right_fit_null, lefty,righty, binary_warped, print_data=True):
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    eval = np.max(np.array([lefty,righty]))
    y_eval = np.max(eval)

    ym_per_pix = 0.5 / 49   # meters per pixel in y dimension
    xm_per_pix = 0.5 / 65 # meter per pixel in x dimension

    # Define left and right lanes in pixels
    if (left_fit_null == 0):
        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        left_lane_bottom = (left_fit[0]) * (y_eval) ** 2 + left_fit[1] * y_eval + left_fit[2]
    else:
        if right_fit_null == 0:
            left_curverad = 1
            left_lane_bottom = (right_fit[0]) * (y_eval) ** 2 + right_fit[1] * (y_eval) + right_fit[2]-1.75/xm_per_pix
            print("here")

        else:
            left_curverad = 10
            left_lane_bottom = 160 - 2/xm_per_pix

    if (right_fit_null == 0):
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        right_curverad = ((1 + (
                    2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        right_lane_bottom = (right_fit[0]) * (y_eval) ** 2 + right_fit[1] * y_eval + right_fit[2]
    else:
        if left_fit_null == 0:
            right_curverad = 1
            right_lane_bottom = (left_fit[0]) * (y_eval-640) ** 2 + left_fit[1] * (y_eval) + left_fit[2] +1.75/xm_per_pix
        else:
            right_curverad = 10
            right_lane_bottom = 160 + 2/xm_per_pix
    lane_center = (left_lane_bottom + right_lane_bottom) / 2.

    print(left_lane_bottom, right_lane_bottom,lane_center,y_eval)
    center_image = 160
    # print(left_lane_bottom, right_lane_bottom, lane_center)
    center = (lane_center - center_image) * xm_per_pix  # Convert to meters
    if print_data == True:
        # Now our radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm', center, 'm')

    return left_curverad, right_curverad, center,


def add_text_to_image(img, left_cur, right_cur, center):
    """
    Draws information about the center offset and the current lane curvature onto the given image.
    :param img:
    """
    cur = (left_cur + right_cur) / 2.
    # print(cur, left_cur, right_cur)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Radius of Curvature = %f(m)' % cur, (50, 50), font, 1, (255, 255, 255), 2)

    left_or_right = "left" if center < 0 else "right"
    cv2.putText(img, 'Vehicle is %.2fm %s of center' % (center, left_or_right), (50, 100), font, 1,
                (255, 255, 255), 2)


def draw_lines(undist, warped, left_fit, left_fit_null, right_fit, right_fit_null, left_cur, right_cur, center, Minv,
               show_img=True):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    # Fit new polynomials to x,y in world space
    if (left_fit_null == 0):
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    else:
        left_fitx = 0 * ploty ** 2 + 0 * ploty + 0
    if (right_fit_null == 0):
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    else:
        right_fitx = 0 * ploty ** 2 + 0 * ploty + 640
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    add_text_to_image(result, left_cur, right_cur, center)
    # print(left_cur, right_cur)
    if show_img == True:
        # plt.figure(figsize=(10,10))
        fig = plt.figure()
        plt.imshow(result)

    return result


def sep_dir(nome):
    # nome = nome[nome.find('_',48,len(nome)):len(nome)]
    nome = nome[nome.find('รง', 0, len(nome)):len(nome)]
    # print("nome, ", nome)
    # direita = nome[1:nome.find('_',3,len(nome))]
    razao = nome[1:nome.find('j', 1, len(nome)) - 1]
    return razao


def dif_speed(left_cur, right_cur, center, vmax):
    lamb = 3
    global vb
    global cur
    if center < -1:
        center = -1
    elif center > 1:
        center = 1
    # vmax = 5
    cur = (left_cur + right_cur) / 2.
    # dc = 320 - center
    # print(type(cur))

    # vb = 0.1*cur + 5
    vb = (vmax * (1 - math.exp(-lamb * abs(cur))))/(1 - math.exp(-lamb))
    # print("Velocidade base: " , vb)
    # if vb > 10:
    #    vb = 10
    #dv = 20 * vb * center
    dv = vb*center/0.21
    # if(cur > 5):
    vleft = vb + dv
    vright = vb - dv
    # else:
    # vleft = 5 + dv/2
    # vright = 5 - dv/2
    if vleft > vb:
        vleft = vb
    elif vleft < 0:
        #print("here")
        vleft = 0
    if vright > vb:
        vright = vb
    if vright < 0:
        vright = 0
    return (vleft, vright, vb)


def angleG(alpha, beta, gamma):
    return (2 * math.acos(math.cos((alpha + gamma) / 2) * math.cos(beta / 2)))


class EKF:
    def __init__(self, vehicleWidth, initialStates, probMatrix, Q):
        # important attributes
        self.vehicleWidth = vehicleWidth
        self.initialStates = initialStates
        self.probVector = []
        self.attributeQ = Q
        self.stdDev = np.array([[],[],[]])
        # matrix for the error model
        self.attributeA = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.attributeC = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.attributeD = np.array([[0, 0], [0, 0], [0, 0]])

        # variable to store states
        self.attributeX = initialStates

        # attribute to store previous probability matrix
        self.attributeP0 = probMatrix
        self.probVector.append(self.attributeP0)
        self.stdDev = np.append(self.stdDev,np.sqrt(np.diag(self.attributeP0)).reshape(3,1),axis = 1)
    def predict(self, odom, dT):
        # useful variables
        x0 = self.initialStates[0,0]
        y0 = self.initialStates[1,0]
        theta0 = self.initialStates[2,0]

        vwidth = self.vehicleWidth
        P0 = self.attributeP0
        Q0 = self.attributeQ

        # continuous system variables
        Ac = self.attributeA
        Cc = self.attributeC
        Bc = np.array(
            [[0.5 * cos(theta0), 0.5 * cos(theta0)], [0.5 * sin(theta0), 0.5 * sin(theta0)], [1 / vwidth, -1 / vwidth]])
        Dc = self.attributeD
        sys_c = control.StateSpace(Ac, Bc, Cc, Dc)

        # discrete system variables
        sys_d = sys_c.sample(dT)
        Ad = sys_d.A
        Bd = sys_d.B
        Cd = sys_d.C
        Dd = sys_d.D

        self.attributeAd = Ad
        self.attributeBd = Bd

        # prediction
        aux_x = x0 + 0.5 * (odom[0] + odom[1]) * cos(theta0)
        #print(x0,aux_x)
        aux_y = y0 + 0.5 * (odom[0] + odom[1]) * sin(theta0)
        aux_theta = theta0 + (odom[0] - odom[1]) / vwidth
       # print(aux_x,aux_y,aux_theta)

        P0 = Ad * P0 * Ad.transpose() + Bd * Q0 * Bd.transpose()

        self.attributeP0 = P0
        self.probVector.append(P0)
        self.attributeX = np.append(self.attributeX, np.array([[aux_x], [aux_y], [aux_theta]]), axis=1)
        self.stdDev = np.append(self.stdDev, np.sqrt(np.diag(self.attributeP0)).reshape(3, 1), axis=1)
        self.initialStates = np.array([[aux_x], [aux_y], [aux_theta]])
        #print("estou")

    def update(self, gps, R, dT):
        # useful variables
        Ad = self.attributeAd
        Bd = self.attributeBd
        Cd = self.attributeC
        Dd = self.attributeD
        P0 = self.attributeP0

        #print("attribute x",self.attributeX[:,self.attributeX.shape[1]-1].reshape(3,1))

        # compute kalman gain
        Gfk = P0 * Cd.transpose() * np.linalg.inv(Cd * P0 * Cd.transpose() + R)
        X_now = self.attributeX[:, self.attributeX.shape[1] - 1].reshape(3,1)

        # compute error
        self.attributedY = X_now- gps

        # adjust with kalman gain
        Xek = np.dot(Gfk, (X_now- gps))


        # store probability matrix
        self.attributeP0 = (np.eye(P0.shape[0],P0.shape[1]) - Gfk * Cd) * P0
        self.probVector.append(self.attributeP0)
        #print("aqui")
        # correct states
        #print((X_now-Xek).reshape(1,3))
        self.attributeX[:,self.attributeX.shape[1]-1] = (X_now - Xek).reshape(1,3)
        self.initialStates=self.attributeX[:,self.attributeX.shape[1]-1].reshape(3,1)
        del X_now

def odom_sep(aux):
    left = aux[aux.find('l', 0, len(aux)):aux.find('r',0,len(aux))]
    right = aux[aux.find('r', 0, len(aux)):aux.find('x',0,len(aux))]
    return [right,left]

def gps_sep(aux):
    x = aux[aux.find('x', 0, len(aux)):aux.find('y',0,len(aux))]
    y = aux[aux.find('y', 0, len(aux)):aux.find('t',0,len(aux))]
    t = aux[aux.find('t', 0, len(aux)):aux.find('s',0,len(aux))]
    return [x,y,t]

def gps_status(aux):
    status = aux[aux.find('s',0,len(aux)):len(aux)]
    return status

def prop_control(img, vmax, show_image = False):
    #print('ini1')
    undist, sxbinary, s_binary, combined_binary1, warped_im, Minv = lane_detector(img)
    #print('ini2')
    left_fit, left_fit_null, right_fit, right_fit_null, lefty, righty, out_img = fit_lines(warped_im,
                                                                                                   plot=False)
    #print('ini3')
    left_cur, right_cur, center = curvature(left_fit, left_fit_null, right_fit, right_fit_null, lefty, righty,
                                                    warped_im,
                                                    print_data=False)
    #print('ini4')
    result = draw_lines(undist, warped_im, left_fit, left_fit_null, right_fit, right_fit_null, left_cur,
                                right_cur, center, Minv, show_img=False)
    #print('ini5')
    lspeed, rspeed, vb = dif_speed(left_cur, right_cur, center, vmax)
    #print('ini6')
    if show_image == True:
        cv2.imshow('Resultado', result)
        cv2.waitKey(10)

    return lspeed,rspeed,vb
