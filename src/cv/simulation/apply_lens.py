from fdlite import FaceDetection, FaceLandmark, face_detection_to_roi
from fdlite import IrisLandmark, iris_roi_from_face_landmarks
from fdlite.examples import iris_recoloring
from fdlite.transform import bbox_from_landmarks
from PIL import Image
import cv2
from skimage.draw import polygon
import numpy as np

class Lens:
	def __init__(self) -> None:
		self.detect_faces = FaceDetection()
		self.detect_face_landmarks = FaceLandmark()
		self.detect_iris = IrisLandmark()
		self.left_eye_results = None
		self.right_eye_results = None

	
	def apply_lens(self, w_frame, r, g, b):
		img = Image.fromarray(w_frame)

		face_detections = self.detect_faces(img)
		if len(face_detections) > 0:
			# get ROI for the first face found
			face_roi = face_detection_to_roi(face_detections[0], img.size)
			# detect face landmarks
			face_landmarks = self.detect_face_landmarks(img, face_roi)
			# get ROI for both eyes
			eye_roi = iris_roi_from_face_landmarks(face_landmarks, img.size)
			left_eye_roi, right_eye_roi = eye_roi
			# detect iris landmarks for both eyes
			self.left_eye_results = left_eye_results = self.detect_iris(img, left_eye_roi)
			self.right_eye_results = right_eye_results = self.detect_iris(img, right_eye_roi, is_right_eye=True)

			# TODO: user skimage polygon insted of bbox

			# l_eye_x = [lmark.x for lmark in left_eye_results.iris]
			# l_eye_y = [lmark.y for lmark in left_eye_results.iris]
			
			# r_eye_x = [lmark.x for lmark in right_eye_results.iris]
			# r_eye_y = [lmark.y for lmark in right_eye_results.iris]

			# l_rr, l_cc = polygon(l_eye_x, r_eye_y)
			# r_rr, r_cc = polygon(r_eye_x, r_eye_y)


			# l_eye_box = bbox_from_landmarks(left_eye_results.iris).absolute(img.size)
			# r_eye_box = bbox_from_landmarks(right_eye_results.iris).absolute(img.size)

			# y_left, x_left = np.mgrid[int(l_eye_box.xmin): int(l_eye_box.xmax): 1, int(l_eye_box.ymin): int(l_eye_box.ymax): 1]
			# y_right, x_right = np.mgrid[int(r_eye_box.xmin): int(r_eye_box.xmax): 1, int(r_eye_box.ymin): int(r_eye_box.ymax): 1]

			# x_left = x_left.flatten()
			# y_left = y_left.flatten()
			# x_right = x_right.flatten()
			# y_right = y_right.flatten()
			
			# self.y_all = np.concatenate((y_left, y_right))
			# self.x_all = np.concatenate((x_left, x_right))

			# change the iris color
			iris_recoloring.recolor_iris(img, left_eye_results, iris_color=(b, g, r))
			iris_recoloring.recolor_iris(img, right_eye_results, iris_color=(b, g, r))

			# self.im_copy = np.array(img)[:, :, ::-1]
			# return self.im_copy

			return np.array(img)
	
	def apply_lens_static(self, w_frame, r, g, b):
		img = Image.fromarray(w_frame)

		# change the iris color
		iris_recoloring.recolor_iris(img, self.left_eye_results, iris_color=(b, g, r))
		iris_recoloring.recolor_iris(img, self.right_eye_results, iris_color=(b, g, r))

		return np.array(img)