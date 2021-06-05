# import cv2
# import src.cv.makeup.utils as mutils

# # mutils.start_cam()
# mutils.start_cam()
# mutils.enable_makeup('eyeshadow', 34, 74, 162, .1)
# # mutils.enable_makeup('blush', 87, 36, 51, .4)
# # mutils.enable_makeup('eyeliner', 142, 30, 29, .8)
# # mutils.enable_makeup('lipstick', 34, 74, 167, .6)
# # mutils.enable_makeup('concealer', 87, 51, 36, 1)
# # mutils.enable_makeup('foundation', 255, 253, 208, .3)
# # mutils.enable_makeup('lens', 74, 136, 237)
# while mutils.Globals.cap.isOpened():
#     cv2.imshow("Frame", mutils.apply_makeup())
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break

import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    im_height, im_width = image.shape[:2]
    if results.detections:
      for detection in results.detections:
        f_xmin = int((detection.location_data.relative_bounding_box.xmin - .1) * im_width)
        f_ymin = int((detection.location_data.relative_bounding_box.ymin - .2) * im_height)
        f_width = int((detection.location_data.relative_bounding_box.width + .2) * im_width)
        f_height = int((detection.location_data.relative_bounding_box.height + .25) * im_height)
        cv2.rectangle(image, (f_xmin, f_ymin), (f_xmin+f_width, f_ymin+f_height), (100, 20, 23))
        # image = image[f_ymin:f_ymin+f_height, f_xmin:f_xmin+f_width]
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
