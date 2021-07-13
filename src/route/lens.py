# import io
# import os
# from flask import request, Blueprint
# from flask_cors import cross_origin
# from PIL import Image
# from base64 import encodebytes
# from src.route.json_encode import JSONEncoder
# from src.cv.makeup import commons, constants
# from mediapipe.python.solutions import face_detection
# from mediapipe.python.solutions import face_mesh
# from skimage.draw import polygon

# from fdlite import FaceDetection, FaceLandmark, face_detection_to_roi
# from fdlite import IrisLandmark, iris_roi_from_face_landmarks
# from fdlite.examples import iris_recoloring
# from PIL import Image
# import numpy as np

# from src.settings import SIMULATOR_INPUT, SIMULATOR_OUTPUT
# import cv2
# import imutils
# from flask import Flask, request, Response


# lens = Blueprint('lens', __name__)

# # load detection models
# detect_faces = FaceDetection()
# detect_face_landmarks = FaceLandmark()
# detect_iris = IrisLandmark()


# # This method executes before any API request


# @lens.before_request
# def before_request():
#     print('Start eyeliner API request')

# # --------------------------------- generl defs

# def get_response_image(image_path):
#     pil_img = Image.open(image_path, mode='r')  # reads the PIL image
#     byte_arr = io.BytesIO()
#     # convert the PIL image to byte array
#     pil_img.save(byte_arr, format='JPEG')
#     encoded_img = encodebytes(byte_arr.getvalue()).decode(
#         'ascii')  # encode as base64
#     return encoded_img

# # ------------------------------------ non general
# @lens.route('/api/makeup/image/lens', methods=['POST'])
# @cross_origin()
# def simulator_lip():
#     # check if the post request has the file part
#     if 'user_image' not in request.files:
#         return {"detail": "No file found"}, 400
#     user_image = request.files['user_image']
#     if user_image.filename == '':
#         return {"detail": "Invalid file or filename missing"}, 400
#     user_id = request.form.get('user_id')
#     image_copy_name = 'simulated_image-{}.jpg'.format(str(user_id))
#     user_image.save(os.path.join(SIMULATOR_INPUT, image_copy_name))    

#     r = int(request.form.get('r_value'))
#     g = int(request.form.get('g_value'))
#     b = int(request.form.get('b_value'))

#     # open image
#     img = Image.open(os.path.join(SIMULATOR_INPUT, image_copy_name))
#     # detect face
#     face_detections = detect_faces(img)
#     if len(face_detections) > 0:
#         # get ROI for the first face found
#         face_roi = face_detection_to_roi(face_detections[0], img.size)
#         # detect face landmarks
#         face_landmarks = detect_face_landmarks(img, face_roi)
#         # get ROI for both eyes
#         eye_roi = iris_roi_from_face_landmarks(face_landmarks, img.size)
#         left_eye_roi, right_eye_roi = eye_roi
#         # detect iris landmarks for both eyes
#         left_eye_results = detect_iris(img, left_eye_roi)
#         right_eye_results = detect_iris(img, right_eye_roi, is_right_eye=True)
#         # change the iris color
#         iris_recoloring.recolor_iris(img, left_eye_results, iris_color=(r, g, b))
#         iris_recoloring.recolor_iris(img, right_eye_results, iris_color=(r, g, b))

#         img = np.array(img)[:, :, ::-1]
#     else:
#         print('no face detected :(')

#     result = []
#     predict_result = save_iamge(img,r,g,b,"lens", 1)
#     [result.append(predict_result) for _ in range(3)]

#     encoded_img = []
#     for image_path in result:
#         encoded_img.append(get_response_image(
#             '{}/{}'.format(SIMULATOR_OUTPUT, image_path)))

#     return (JSONEncoder().encode(encoded_img), 200)


# def save_iamge(img,r_value,g_value,b_value,makeup_type,intensity):
#     name = 'color_' + str(r_value) + '_' + \
#     str(g_value) + '_' + str(b_value)
#     file_name = '{}_output-{}_{}.jpg'.format(makeup_type,intensity, name)
#     cv2.imwrite(os.path.join(SIMULATOR_OUTPUT, file_name), img)
#     return file_name