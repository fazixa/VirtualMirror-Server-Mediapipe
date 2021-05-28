from fdlite import FaceDetection, FaceLandmark, face_detection_to_roi
from fdlite import IrisLandmark, iris_roi_from_face_landmarks
from fdlite.examples import iris_recoloring
from fdlite.transform import bbox_from_landmarks
from PIL import Image
import cv2
import numpy as np

class Lens:
	def __init__(self) -> None:
		self.detect_faces = FaceDetection()
		self.detect_face_landmarks = FaceLandmark()
		self.detect_iris = IrisLandmark()
		self.x_all = []
		self.y_all = []
		self.im_copy = None

	
	def apply_lens(self, w_frame, r, g, b):
		img = cv2.cvtColor(w_frame, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(img)

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
			left_eye_results = self.detect_iris(img, left_eye_roi)
			right_eye_results = self.detect_iris(img, right_eye_roi, is_right_eye=True)

			
			l_eye_box = bbox_from_landmarks(left_eye_results.iris).absolute(img.size)
			r_eye_box = bbox_from_landmarks(right_eye_results.iris).absolute(img.size)

			y_left, x_left = np.mgrid[int(l_eye_box.xmin): int(l_eye_box.xmax): 1, int(l_eye_box.ymin): int(l_eye_box.ymax): 1]
			y_right, x_right = np.mgrid[int(r_eye_box.xmin): int(r_eye_box.xmax): 1, int(r_eye_box.ymin): int(r_eye_box.ymax): 1]

			x_left = x_left.flatten()
			y_left = y_left.flatten()
			x_right = x_right.flatten()
			y_right = y_right.flatten()
			
			self.y_all = np.concatenate((y_left, y_right))
			self.x_all = np.concatenate((x_left, x_right))

			# change the iris color
			iris_recoloring.recolor_iris(img, left_eye_results, iris_color=(b, g, r))
			iris_recoloring.recolor_iris(img, right_eye_results, iris_color=(b, g, r))

			self.im_copy = np.array(img)[:, :, ::-1]
			return self.im_copy