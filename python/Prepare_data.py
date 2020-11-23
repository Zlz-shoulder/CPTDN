# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 08:45:41 2020

@author: Zhong.Lianzhen
"""
import tensorflow as tf
import numpy as np
import scipy.io
import cv2
from scipy import signal
import os
import matplotlib.pyplot as plt

img_width = 128
img_heigh = 128
main_path = "/T3N1_data/"

#center jitter
def shuffe_center(x_center,y_center,order = 0):
    if order == 0:
        return x_center,y_center
        # print('No shuffe_center')
    if order == 1:
        x_center -= 5
        return x_center,y_center
    if order == 2:
        x_center += 5
        return x_center,y_center
    if order == 3:
        y_center -= 5
        return x_center,y_center
    if order == 4:
        y_center += 5
    
    return x_center,y_center

#grayvalue jitter
def shuffe_grayValue(v0):
    alpha = np.random.randint(3) - 1
    v0 = v0 - alpha*5
    return v0


#rotate
def rotate_center(slice0, slice1, angle = 5):

    nn = 5 #Number of rotation angles: -10,-5,0,5,10
    rows,cols = slice0.shape
    rotate = np.random.randint(nn)-nn//2
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle*rotate, 1)

    slice0 = cv2.warpAffine(slice0, M, (rows, cols),cv2.INTER_LINEAR)
    slice1 = cv2.warpAffine(slice1, M, (rows, cols),cv2.INTER_MAX)

    return slice0,slice1

#Gaussian smoothing
def gaussBlur(img, sigma, H, W, _boundary='fill', _fillvalue=0):

    gaussKernel_x = cv2.getGaussianKernel(W, sigma, cv2.CV_32F)
    gaussKernel_x = np.transpose(gaussKernel_x)

    gaussBlur_x = signal.convolve2d(img, gaussKernel_x, mode="same",
                                    boundary=_boundary, fillvalue=_fillvalue)

    gaussKernel_y = cv2.getGaussianKernel(H, sigma, cv2.CV_32F)
    gaussBlur_xy = signal.convolve2d(gaussBlur_x, gaussKernel_y, mode="same",
                                     boundary=_boundary, fillvalue=_fillvalue)
    return gaussBlur_xy

def data_aug(v0,v1):
    v0 = v0.astype(np.float32)
#    rotate
    v0,v1 = rotate_center(v0,v1)
#    grayvalue jitter
    v0 = shuffe_grayValue(v0)
#    smooth_gaussian
    v0 = gaussBlur(v0,1,7,7)
#    center jitter
    v0 = v0.astype(np.float32)
    v0 = np.clip(v0, 0.0, 2048.0)
    v0 = v0/2048.0
    x_coord,y_coord = np.where(v1 != 0)
    x_min = np.min(x_coord)
    x_max = np.max(x_coord)
    y_min = np.min(y_coord)
    y_max = np.max(y_coord)
    x_center = int((x_min+x_max)/2)
    y_center = int((y_min+y_max)/2)
    x_center,y_center = shuffe_center(x_center, y_center, order = np.random.randint(5))
    x_start = int(x_center-img_width/2)
    y_start = int(y_center-img_heigh/2)
    v0 = v0[x_start:x_start+img_width,y_start:y_start + img_heigh]
    v1 = v1[x_start:x_start+img_width,y_start:y_start + img_heigh]

    v1 = v1.astype(np.float32)
    #build a sample
    img = np.stack([v0,v1],axis = 0)
    img = img.transpose(2,1,0)

    return img

def preprocess(pat_ID,sequence, is_train = True):
    input_img = []
    pat_path = pat_ID + '_' + sequence + '.mat'
    data = scipy.io.loadmat(os.path.join(main_path,pat_path))
    # print(pat_path)
    v_o,v_s = data['v_o'], data['v_s']
    num_slice = v_o.shape[-1]
    if is_train:
        for i in range(num_slice):
            v0 = v_o[:,:,i]
            v1 = v_s[:,:,i]
            v1[v1 != 0] = 1
            img = data_aug(v0,v1)
            #build a mini-batch
            input_img.append(img)
    else:
        for i in range(num_slice):
            v0 = v_o[:,:,i]
            v1 = v_s[:,:,i]
            v1[v1 != 0] = 1
            v0 = v0.astype(np.float32)
            #Data standardization
            v0 = np.clip(v0, 0.0, 2048)
            v0 = v0/2048
            #get the crop at the center
            x_coord,y_coord = np.where(v1 != 0)
            x_min = np.min(x_coord)
            x_max = np.max(x_coord)
            y_min = np.min(y_coord)
            y_max = np.max(y_coord)
            x_center = int((x_min+x_max)/2)
            y_center = int((y_min+y_max)/2)
            x_start = int(x_center-img_width/2)
            y_start = int(y_center-img_heigh/2)
            v0 = v0[x_start:x_start+img_width,y_start:y_start + img_heigh]
            v1 = v1[x_start:x_start+img_width,y_start:y_start + img_heigh]
            v1 = v1.astype(np.float32)
            #build a sample
            img = np.stack([v0,v1],axis = 0)
            img = img.transpose(2,1,0)
            #build a mini-batch
            input_img.append(img)
    input_img = np.stack(input_img)

    return input_img