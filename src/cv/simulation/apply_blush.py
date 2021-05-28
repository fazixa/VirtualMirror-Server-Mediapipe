import cv2
from pylab import *
import numpy as np
from numpy import c_
from skimage import color

from scipy.interpolate import interp1d, splprep, splev


class Blush(object):

    def __init__(self):
        self.r = 0
        self.g = 0
        self.b = 0
        self.intensity = 0
        self.blush_radius = 1.02
        # Original image

        # All the changes will be applied to im_copy


        self.x_all = []
        self.y_all = []

    def get_boudary(self, landmarks_x, landmarks_y):
        center_right_cheek = np.empty([2], dtype=int)
        cenetr_left_cheek = np.empty([2], dtype=int)
        # right cheek
        r_right_cheek = (landmarks_x[15] - landmarks_x[35]) / 3
        center_right_cheek[0] = (landmarks_x[15] + landmarks_x[35]) / 2.0
        center_right_cheek[1] = (landmarks_y[15] + landmarks_y[35]) / 2.0
        # left cheeck
        r_left_cheeck = (landmarks_x[1] - landmarks_x[31]) / 3.5
        cenetr_left_cheek[0] = (landmarks_x[1] + landmarks_x[31]) / 2.0
        cenetr_left_cheek[1] = (landmarks_y[1] + landmarks_y[31]) / 2.0

        return r_right_cheek, center_right_cheek, r_left_cheeck, cenetr_left_cheek

    def fill(self, r, center):
        points_1 = [center[0] - r, center[1]]
        points_2 = [center[0], center[1] - r]
        points_3 = [center[0] + r, center[1]]
        points_4 = [center[0], center[1] + r]
        points_5 = points_1

        points = np.array([points_1, points_2, points_3, points_4, points_5])

        x, y = points[0:5, 0], points[0:5, 1]

        tck, u = splprep([x, y], s=0, per=1)
        unew = np.linspace(u.min(), u.max(), 1000)
        xnew, ynew = splev(unew, tck, der=0)
        tup = c_[xnew.astype(int), ynew.astype(int)].tolist()
        coord = list(set(tuple(map(tuple, tup))))
        coord = np.array([list(elem) for elem in coord])
        return np.array(coord[:, 0], dtype=np.int32), np.array(coord[:, 1], dtype=np.int32)

    def get_interior_points(self, x, y):
        intx = []
        inty = []
        print('start get_interior_points')

        def ext(a, b, i):
            a, b = round(a), round(b)
            intx.extend(arange(a, b, 1).tolist())
            inty.extend((ones(b - a) * i).tolist())

        x, y = np.array(x), np.array(y)
        print('x,y get_interior_points')
        xmin, xmax = amin(x), amax(x)
        xrang = np.arange(xmin, xmax + 1, 1)
        print(type(xrang))
        print('x-rang')
        print(xrang)
        for i in xrang:
            try:
                ylist = y[where(x == i)]
                ext(amin(ylist), amax(ylist), i)
            except ValueError:  # raised if `y` is empty.
                pass

        print('xrang2 get_interior_points')
        return np.array(intx, dtype=np.int32), np.array(inty, dtype=np.int32)


    def apply_blush(self, img, landmarks_x, landmarks_y, r, g, b, intensity):
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)
        self.intensity = intensity
        self.image = img
        self.im_copy = img.copy()
        self.height, self.width = self.image.shape[:2]

        r_right_cheek, center_right_cheek, r_left_cheek, center_left_cheek = self.get_boudary(landmarks_x, landmarks_y)

        x_right, y_right = self.fill(r_right_cheek, center_right_cheek)
        x_left, y_left = self.fill(r_left_cheek, center_left_cheek)

        x_right_all, y_right_all = self.get_interior_points(x_right, y_right)
        x_left_all, y_left_all = self.get_interior_points(x_left, y_left)
        # x_all = x_left + x_right
        # y_all = y_left + y_right
        self.y_all = np.concatenate((y_left_all, y_right_all))
        self.x_all = np.concatenate((x_left_all, x_right_all))

        self.apply_color(self.x_all, self.y_all )
        self.apply_blur(self.x_all, self.y_all )
        # self.blush(x_right, y_right, x_left, y_left)

        return self.im_copy

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

    def apply_blur(self, x, y):
        # gussian blur
        filter = np.zeros((self.height, self.width))
        cv2.fillConvexPoly(filter, np.array(c_[y, x], dtype='int32'), 1)
        filter = cv2.GaussianBlur(filter, (51, 51), 0)
        # Erosion to reduce blur size
        kernel = np.ones((15,15), np.uint8)
        filter = cv2.erode(filter, kernel, iterations=1)
        alpha = np.zeros([self.height, self.width, 3], dtype='float64')
        alpha[:, :, 0] = filter
        alpha[:, :, 1] = filter
        alpha[:, :, 2] = filter
        self.im_copy = (alpha * self.im_copy + (1 - alpha) * self.image).astype('uint8')
