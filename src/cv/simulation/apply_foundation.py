#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from __future__ import division
from itertools import zip_longest
# import scipy.interpolate
from scipy.interpolate import interp1d
import cv2
import numpy as np
from skimage import color
from PIL import Image
from imutils import face_utils
import os
import dlib
from pylab import *
from skimage import io
import time
from scipy import interpolate
from skimage.io.collection import imread_collection_wrapper


class Foundation(object): 

    def __init__(self):
        self.r = 0
        self.g = 0
        self.b = 0
        self.intensity = 0
        self.eyeshadow_height = 1.02
        # Original image
       
        # All the changes will be applied to im_copy
        self.x_all = []
        self.y_all = []


    def apply_foundation(self, img, landmark_x, landmark_y, landmark_x68,landmark_y68, r_value, g_value, b_value, intensity, ksize_h, ksize_w):
        self.r = int(r_value)
        self.g = int(g_value)
        self.b = int(b_value)
        self.image = img
        self.intensity = intensity


        # shape = self.get_cheek_shape(gray_image)

        self.height, self.width = self.image.shape[:2]
        self.im_copy = img.copy()
        # cv2.imshow("i",self.im_copy)
        # cv2.waitKey

        start = time.time()
        face_top_x = np.r_[landmark_x68[45], landmark_x68[46], landmark_x68[47] , 
            landmark_x68[42] , landmark_x68[39] ,landmark_x68[40] , 
            landmark_x68[41], landmark_x68[36],landmark_x68[1:16]]
        
        face_top_x_81 = np.r_[landmark_x[21],landmark_x[19], landmark_x[75]
        , landmark_x[68], landmark_x[69], landmark_x[70], landmark_x[71], landmark_x[72]
        , landmark_x[73], landmark_x[79], landmark_x[74], landmark_x[22], landmark_x[24]]

        face_top_y = np.r_[ landmark_y68[45], landmark_y68[46], landmark_y68[47],
             landmark_y68[42], landmark_y68[39] , landmark_y68[40] ,
             landmark_y68[41], landmark_y68[36], landmark_y68[1:16]]
        
        face_top_y_81 = np.r_[landmark_y[21], landmark_y[19], landmark_y[75]
        , landmark_y[68], landmark_y[69],landmark_y[70],landmark_y[71],landmark_y[72]
        , landmark_y[73], landmark_y[79],landmark_y[74], landmark_y[22], landmark_y[24]]


        face_top_x, face_top_y = self.get_boundary_points(
           face_top_x, face_top_y)

        face_top_x_81, face_top_y_81 = self.get_boundary_points(
           face_top_x_81, face_top_y_81)


        face_top_y, face_top_x = self.get_interior_points(
            face_top_x, face_top_y)
    ###############################################################3
        face_top_y_81, face_top_x_81 = self.get_interior_points(
            face_top_x_81, face_top_y_81)

        self.y_all = np.concatenate((face_top_x, face_top_x_81)) 
        self.x_all = np.concatenate((face_top_y, face_top_y_81))
        # upper_x = np.r_[landmark_x68[14],
        #      landmark_x68[45], landmark_x68[46], landmark_x68[47],
        #      landmark_x68[42],  landmark_x68[27], landmark_x68[39] , landmark_x68[40] ,
        #      landmark_x68[41], landmark_x68[36], landmark_x68[2]]


        # upper_y = np.r_[landmark_y68[14],
        #      landmark_y68[45], landmark_y68[46], landmark_y68[47],
        #      landmark_y68[42],  landmark_y68[27], landmark_y68[39] , landmark_y68[40] ,
        #      landmark_y68[41], landmark_y68[36], landmark_y68[2]]

        # lower_x = np.r_[ landmark_x68[3:13]]
        # lower_y = np.r_[ landmark_y68[3:13]]

        # lower_path = self.interpolate(lower_x, lower_y, 'cubic')
        # upper_path = self.interpolate(upper_x, upper_y, 'cubic')

        # face_top_y, face_top_x = self.fill(lower_path, upper_path)



        end = time.time()
        print(end-start)
        # for x, y in zip(self.x_all, self.y_all):
        #      self.im_copy = cv2.circle(img, (y, x), 1, (0, 0, 255), -1)

        # self.apply_color(self.x_all, self.y_all )
        self.apply_blur2(self.x_all,self.y_all )


        return self.im_copy

    def interpolate(self, lx=[], ly=[], k1='quadratic'):
        unew = np.arange(lx[0], lx[-1] + 1, 1)
        f2 = interp1d(lx, ly, kind=k1)
        return (f2, unew)

        
    def fill(self, lower_path, upper_path):
        x = []
        y = []
        for i in lower_path[1]:
            for j in range(int(upper_path[0](i)), int(lower_path[0](i))):
                x.append(int(j))
                y.append(int(i))
        return x, y

    def get_boundary_points(self, x, y):
        tck, u = interpolate.splprep([x, y], s=0, per=1)
        unew = np.linspace(u.min(), u.max(), 1500)
        xnew, ynew = interpolate.splev(unew, tck, der=0)
        tup = c_[xnew.astype(int), ynew.astype(int)].tolist()
        coord = list(set(tuple(map(tuple, tup))))
        coord = np.array([list(elem) for elem in coord])
        return np.array(coord[:, 0], dtype=np.int32), np.array(coord[:, 1], dtype=np.int32)

    def get_interior_points(self, x, y):
        intx = []
        inty = []

        def ext(a, b, i):
            a, b = round(a), round(b)
            intx.extend(arange(a, b, 1).tolist())
            inty.extend((ones(b - a) * i).tolist())

        x, y = np.array(x), np.array(y)
        xmin, xmax = amin(x), amax(x)
        xrang = np.arange(xmin, xmax + 1, 1)
   
        for i in xrang:
            try:
                ylist = y[where(x == i)]
                ext(amin(ylist), amax(ylist), i)
            except ValueError:  # raised if `y` is empty.
                pass

        return np.array(intx, dtype=np.int32), np.array(inty, dtype=np.int32)

    def apply_color(self, x, y):
       
        # converting desired parts of the original image to LAB color space
        lip_LAB = color.rgb2lab((self.im_copy[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)
        # calculating mean of each channel
        L, A, B = mean(lip_LAB[:, 0]), mean(lip_LAB[:, 1]), mean(lip_LAB[:, 2])
        # converting the color of the makeup to LAB
        L1, A1, B1 = color.rgb2lab(np.array((self.r / 255., self.g / 255., self.b / 255.)).reshape(1, 1, 3)).reshape(
            3, )
        # applying the makeup color on image
        # L1, A1, B1 = color.rgb2lab(np.array((self.r / 255., self.g / 255., self.b / 255.)).reshape(1, 1, 3)).reshape(3, )
        G = L1 / L
        lip_LAB = lip_LAB.reshape(len(x), 1, 3)
        lip_LAB[:, :, 1:3] = self.intensity * np.array([A1, B1]) + (1 - self.intensity) * lip_LAB[:, :, 1:3]
        lip_LAB[:, :, 0] = lip_LAB[:, :, 0] * (1 + self.intensity * (G - 1))
        # converting back toRGB
        # print(self.r,self.g,self.b)
        self.im_copy[x, y] = color.lab2rgb(lip_LAB).reshape(len(x), 3) * 255
        # cv2.imshow("f", self.im_copy)
        # cv2.waitKey(0)

    def apply_blur(self, x, y):
        # gussian blur
        filter = np.zeros((self.height, self.width))



        cv2.fillConvexPoly(filter, np.array(c_[y, x], dtype='int32'), 1)
        filter = cv2.GaussianBlur(filter, (81, 81), 0) * self.intensity
        # Erosion to reduce blur size
        kernel = np.ones((25,25), np.uint8)
        filter = cv2.dilate(filter, kernel, iterations=1)


        
        alpha = np.zeros([self.height, self.width, 3], dtype='float64')
        alpha[:, :, 0] = filter
        alpha[:, :, 1] = filter
        alpha[:, :, 2] = filter

        # cv2.imshow("img", filter)
        # cv2.waitKey(0)
        self.im_copy = (alpha * self.im_copy + (1 - alpha) * self.image).astype('uint8')


    def apply_blur2(self, x, y):
        intensity = self.intensity
        # Create blush shape
        mask = np.zeros((self.height, self.width))
        cv2.fillConvexPoly(mask, np.array(c_[y, x], dtype='int32'), 1)
     
        mask = cv2.GaussianBlur(mask, (81, 81), 0) * intensity
        kernel = np.ones((35, 35), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        # print(np.array(c_[x_right, y_right])[:, 0])
        val = cv2.cvtColor(self.im_copy, cv2.COLOR_RGB2LAB).astype(float)

        val[:, :, 0] = val[:, :, 0] / 255. * 100.
        val[:, :, 1] = val[:, :, 1] - 128.
        val[:, :, 2] = val[:, :, 2] - 128.
        print(self.r)
        LAB = color.rgb2lab(np.array((float(self.r) / 255., float(self.g) / 255., float(self.b) / 255.)).reshape(1, 1, 3)).reshape(3, )
        mean_val = np.mean(np.mean(val, axis=0), axis=0)


        mask = np.array([mask, mask, mask])
        mask = np.transpose(mask, (1, 2, 0))

        lab = np.multiply((LAB - mean_val), mask)

        val[:, :, 0] = np.clip(val[:, :, 0] + lab[:, :, 0], 0, 100)
        val[:, :, 1] = np.clip(val[:, :, 1] + lab[:, :, 1], -127, 128)
        val[:, :, 2] = np.clip(val[:, :, 2] + lab[:, :, 2], -127, 128)

        self.im_copy = (color.lab2rgb(val) * 255).astype(np.uint8)


    def apply_blur_only(self, new_img, x, y):
        intensity = self.intensity
        # Create blush shape
        mask = np.zeros((self.height, self.width))
        cv2.fillConvexPoly(mask, np.array(c_[y, x], dtype='int32'), 1)
     
        mask = cv2.GaussianBlur(mask, (81, 81), 0) * intensity
        kernel = np.ones((35, 35), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        # print(np.array(c_[x_right, y_right])[:, 0])
        val = cv2.cvtColor(new_img, cv2.COLOR_RGB2LAB).astype(float)

        val[:, :, 0] = val[:, :, 0] / 255. * 100.
        val[:, :, 1] = val[:, :, 1] - 128.
        val[:, :, 2] = val[:, :, 2] - 128.
        print(self.r)
        LAB = color.rgb2lab(np.array((float(self.r) / 255., float(self.g) / 255., float(self.b) / 255.)).reshape(1, 1, 3)).reshape(3, )
        mean_val = np.mean(np.mean(val, axis=0), axis=0)


        mask = np.array([mask, mask, mask])
        mask = np.transpose(mask, (1, 2, 0))

        lab = np.multiply((LAB - mean_val), mask)

        val[:, :, 0] = np.clip(val[:, :, 0] + lab[:, :, 0], 0, 100)
        val[:, :, 1] = np.clip(val[:, :, 1] + lab[:, :, 1], -127, 128)
        val[:, :, 2] = np.clip(val[:, :, 2] + lab[:, :, 2], -127, 128)

        return (color.lab2rgb(val) * 255).astype(np.uint8)