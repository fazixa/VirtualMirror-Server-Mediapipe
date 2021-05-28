from itertools import zip_longest
# import scipy.interpolate
from scipy.interpolate import interp1d
import cv2
import numpy as np
from skimage import color
from PIL import Image
from src.cv.detector import DetectLandmarks
from src.settings import SIMULATOR_INPUT, SIMULATOR_OUTPUT
import os
import dlib
from pylab import *
from skimage import io


class Eyeshadow(object):

    def __init__(self):
        self.r = 0
        self.g = 0
        self.b = 0
        self.intensity = 0
        self.eyeshadow_height = 1.04
        self.image = 0
        # All the changes will be applied to im_copy
        self.im_copy = 0
        self.width = 0
        self.height = 0
        self.x_all = []
        self.y_all = []

    def apply_eyeshadow(self, image, landmarks_x, landmarks_y, rlips, glips, blips, intensity):
        """
        Applies lipstick on an input image.
        ___________________________________
        Args:
            1. `filename (str)`: Path for stored input image file.
            2. `red (int)`: Red value of RGB colour code of lipstick shade.
            3. `blue (int)`: Blue value of RGB colour code of lipstick shade.
            4. `green (int)`: Green value of RGB colour code of lipstick shade.
        Returns:
            `filepath (str)` of the saved output file, with applied lipstick.
        """
        self.r = int(rlips)
        self.g = int(glips)
        self.b = int(blips)
        self.intensity = intensity

        self.image = image
        self.height, self.width = self.image.shape[:2]

        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.im_copy = self.image.copy()
        # self.im_copy = cv2.cvtColor(self.im_copy, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('./data/results/eyeshadow5.jpg',self.im_copy)

        # landmarks_x,landmarks_y=self.get_landmarks(self.image)

        # get eyes point
        x_left_eye_lower, x_left_eye_upper, y_left_eye_lower, y_left_eye_upper, x_right_eye_lower, \
        x_rigth_eye_upper, y_right_eye_lower, y_right_eye_upper = self.get_point(landmarks_x, landmarks_y)
        # print(x_left_eye_lower,"----------------------------------------")
        # create interpolated path
        left_eye_lower_path = self.interpolatex(x_left_eye_lower[:], y_left_eye_lower[:], 'cubic')
        left_eye_upper_path = self.interpolatex(x_left_eye_upper[:], y_left_eye_upper[:], 'cubic')
        right_eye_lower_path = self.interpolatex(x_right_eye_lower[:], y_right_eye_lower[:], 'cubic')
        right_eye_upper_path = self.interpolatex(x_rigth_eye_upper[:], y_right_eye_upper[:], 'cubic')
        # print(left_eye_lower_path)
        # filling points eyelids
        x_all_left, y_all_left = self.fill(left_eye_lower_path, left_eye_upper_path)
        # print(len(x_all_left))
        x_all_right, y_all_right = self.fill(right_eye_lower_path, right_eye_upper_path)
        x_all = x_all_left + x_all_right
        y_all = y_all_left + y_all_right

        self.x_all = x_all
        self.y_all = y_all

        # apply color
        self.apply_color(x_all, y_all)
        self.apply_blur(x_all, y_all)

        # self.im_copy = cv2.cvtColor(self.im_copy, cv2.COLOR_BGR2RGB)

        return self.im_copy

    # -------- general methods ----------
    def interpolatex(self, lx=[], ly=[], k1='quadratic'):
        unew = np.arange(lx[0], lx[-1] + 1, 1)
        f2 = interp1d(lx, ly, kind=k1)
        return (f2, unew)

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

        # self.im_copy = cv2.cvtColor(self.im_copy, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('./eyeshadow2.jpg', self.im_copy)

    def apply_blur(self, x, y):
        # gussian blur
        filter = np.zeros((self.height, self.width))
        cv2.fillConvexPoly(filter, np.array(c_[y, x], dtype='int32'), 1)
        filter = cv2.GaussianBlur(filter, (41, 41), 0)
        # Erosion to reduce blur size
        kernel = np.ones((22, 22), np.uint8)
        filter = cv2.erode(filter, kernel, iterations=1)
        alpha = np.zeros([self.height, self.width, 3], dtype='float64')
        alpha[:, :, 0] = filter
        alpha[:, :, 1] = filter
        alpha[:, :, 2] = filter
        self.im_copy = (alpha * self.im_copy + (1 - alpha) * self.image).astype('uint8')

    def fill(self, lower_path, upper_path):
        xz = []
        yz = []
        # print(upper_path[0],upper_path[1])
        for i in lower_path[1]:
            # print(i)
            for j in range(int(upper_path[0](i)), int(lower_path[0](i))):
                xz.append(int(j))
                yz.append(int(i))
        return xz, yz

    # -------- non general methods ----------
    def get_point(self, x, y):

        # left eye
        x_left_eye_lower = np.r_[x[17], x[36:40]]
        x_left_eye_upper = np.r_[x[17], x[18:22]]
        y_left_eye_lower = np.r_[y[17], y[36:40]]
        y_left_eye_upper = np.r_[y[17], (np.array(y[18:22]) * self.eyeshadow_height)]

        # right eye
        x_right_eye_lower = np.r_[x[42:46], x[26]]
        x_rigth_eye_upper = np.r_[x[22:27]]
        y_right_eye_lower = np.r_[y[42:46], y[26]]
        y_right_eye_upper = np.r_[(np.array(y[22:27]) * self.eyeshadow_height)]

        return x_left_eye_lower, x_left_eye_upper, y_left_eye_lower, y_left_eye_upper, \
               x_right_eye_lower, x_rigth_eye_upper, y_right_eye_lower, y_right_eye_upper

    # def get_landmarks(self,img):

    #     detected_faces = detector(img, 0)

    #     pose_landmarks = face_pose_predictor(img, detected_faces[0])
    #     # preparing landmarks
    #     landmarks_x = []
    #     landmarks_y = []
    #     for i in range(68):
    #         landmarks_x.append(pose_landmarks.part(i).x)
    #         landmarks_y.append(pose_landmarks.part(i).y)
    #     # print(landmarks_x)
    #     return landmarks_x,landmarks_y
