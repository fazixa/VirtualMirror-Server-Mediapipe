import cv2
import dlib
import os
import numpy as np
from pylab import *
from scipy import interpolate
from imutils import face_utils
from os.path import join as pjoin
from src.settings import COLOR_PREDICTION_INPUT, COLOR_PREDICTION_OUTPUT, SHAPE_68_PATH,SHAPE_81_PATH, SAVE_FILE_TYPE


PREDICTOR_PATH = SHAPE_81_PATH


class DetectLandmarks(object):

    IMAGE_DATA = 'IMAGE_DATA'
    FILE_READ = 'FILE_READ'
    NETWORK_BYTE_STREAM = 'NETWORK_BYTE_STREAM'
    COLOR_PREDICTION_FILE_READ = "COLOR_PREDICTION_FILE_READ"

    def __init__(self):
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)
        self.detector = dlib.get_frontal_face_detector()

    def get_landmarks(self, image):
        """ Extract the landmarks from a given image.
        Returns `None` if no landmarks found.
        """
        try:
            rects = self.detector(image, 1)
            size = len(rects)
            image = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2GRAY)
            grayImg = cv2.equalizeHist(image)
            if size == 0:
                return None, None
            return np.matrix([[p.x, p.y] for p in self.predictor(grayImg, rects[0]).parts()])
        except Exception:
            return None

    def get_face_data(self, image_file, flag=None):
        """
        Returns all facial landmarks in a given image.
        """
        image = 0
        if flag == self.FILE_READ:
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif flag == self.NETWORK_BYTE_STREAM:
            image = cv2.imdecode(
                np.fromstring(image_file.read(),
                              np.uint8), cv2.IMREAD_UNCHANGED
            )
        elif flag == self.IMAGE_DATA or flag is None:
            image = image_file
            landmarks = self.get_landmarks(image)
        elif flag == self.COLOR_PREDICTION_FILE_READ:
            image = cv2.imread(image_file)
            landmarks = self.get_landmarks(image)
        if landmarks[0] is None or landmarks[1] is None:
            return None
        return landmarks

    def save_localize_lanmark_image(self, image, output_path, output_name, flag=None):
        if flag == self.COLOR_PREDICTION_FILE_READ:
            image_data = cv2.imread(image)
            image_copy = np.copy(image_data)
        else:
            image_copy = np.copy(image)
        lanmarks = self.get_face_data(image_file=image, flag=flag)
        for (i, point) in enumerate(lanmarks):
            x = point[0, 0]
            y = point[0, 1]
            cv2.circle(image_copy, (x, y), 2, (0, 255, 255), 2)
            cv2.putText(image_copy, "{}".format(i+1), (x, y-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(pjoin(output_path, "".join(
            (output_name, SAVE_FILE_TYPE))), image_copy)

    def convert_request_files_to_image(self, request_file):
        return cv2.imdecode(np.fromstring(request_file.read(), np.uint8), cv2.IMREAD_COLOR)

    def save_file(self, output_path, image, filename):
        cv2.imwrite(pjoin(output_path, filename), image)

    def create_box(self, image, output_path, output_name, np_points, flag=None):
        if flag == self.COLOR_PREDICTION_FILE_READ:
            image = cv2.imread(image)
        mask = np.zeros_like(image)
        # Our feature
        mask = cv2.fillPoly(mask, [np_points], (255, 255, 255))
        # cv2.imwrite(pjoin(output_path,"Mask.jpg"), mask)
        image = cv2.bitwise_and(image, mask)
        # cv2.imwrite(pjoin( APP_OUTPUT,"OnlyLipIMG.jpg"), img)
        bbox = cv2.boundingRect(np_points)
        x, y, w, h = bbox
        image_crop = image[y: y + h, x: x + w]
        # print("Crop=",image_crop)
        cv2.imwrite(pjoin(output_path, "".join(
            (output_name, SAVE_FILE_TYPE))), image_crop)
        # imgCrop = cv2.resize(imgCrop, (0,0), None, scale, scale)
        # return imgCrop

    def get_cheek_np(self, image, flag=None):
        """
        Return points for cheek as np array in given image
        """
        lanmarks = self.get_face_data(image_file=image, flag=flag)
        np_list = np.concatenate(
            (lanmarks[2], lanmarks[3], lanmarks[4], lanmarks[5], lanmarks[49], lanmarks[29], lanmarks[40]))
        np_list1 = np.concatenate(
            (lanmarks[11], lanmarks[12], lanmarks[13], lanmarks[14], lanmarks[15], lanmarks[35], lanmarks[53], lanmarks[54]))

        print(np_list1)
        return np_list

    def get_lip_np(self, image, flag=None):
        """
        Returns points for lips as np array in given image.
        """
        lanmarks = self.get_face_data(image_file=image, flag=flag)
        return lanmarks[48:]

    def get_lips(self, image_file, flag=None):
        """
        Returns points for lips in given image.
        """
        landmarks = self.get_face_data(image_file, flag)
        if landmarks is None:
            return None
        lips = ""
        for point in landmarks[48:]:
            lips += str(point).replace('[', '').replace(']', '') + '\n'
        return lips

    def get_boundary_points(self, x, y):
        tck, u = interpolate.splprep([x, y], s=0, per=1)
        unew = np.linspace(u.min(), u.max(), 1000)
        xnew, ynew = interpolate.splev(unew, tck, der=0)
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

    def get_cheek_shape(self, gray_image):
        faces = self.detector(gray_image, 1)
        shape = self.predictor(gray_image, faces[0])
        shape = face_utils.shape_to_np(shape)
        shape = shape.tolist()
        for i, j in enumerate(shape):
            shape[i] = (j[0], j[1])
        return shape
