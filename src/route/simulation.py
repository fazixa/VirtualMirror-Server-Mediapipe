# import io
# from skimage import io as skimage_io
# import os
from flask import Flask, request, send_from_directory, jsonify, Blueprint
from flask_cors import cross_origin
# from PIL import Image
# from base64 import encodebytes
# # from src.tints.utils.json_encode import JSONEncoder
# # from src.cv.simulation.apply_makeup import ApplyMakeup
# from src.cv.simulation.apply_eyeshadow import Eyeshadow
# from src.settings import SIMULATOR_INPUT, SIMULATOR_OUTPUT
# import cv2
# import time
# import imutils
from flask import Flask, render_template, url_for, request, Response
# import dlib
# import threading

import src.cv.makeup.utils as mutils

# from src.settings import SHAPE_68_PATH

simulation = Blueprint('simulation', __name__)
# outputFrame = None

# # detector = dlib.get_frontal_face_detector()
# # face_pose_predictor = dlib.shape_predictor(SHAPE_68_PATH)


# # This method executes before any API request


@simulation.before_request
def before_request():
    print('Start Simulation API request')


# # ------------------------------ depracated ----------------------
# @simulation.route('/api/test/simulation')
# def test_simulation():
#     return "Success get simulation api call", 200


# def get_response_image(image_path):
#     pil_img = Image.open(image_path, mode='r')  # reads the PIL image
#     byte_arr = io.BytesIO()
#     # convert the PIL image to byte array
#     pil_img.save(byte_arr, format='JPEG')
#     encoded_img = encodebytes(byte_arr.getvalue()).decode(
#         'ascii')  # encode as base64
#     return encoded_img


# @simulation.route('/api/simulator/lip', methods=['POST'])
# @cross_origin()
# def simulator_lip():
#     # check if the post request has the file part
#     if 'user_image' not in request.files:
#         return {"detail": "No file found"}, 400
#     user_image = request.files['user_image']
#     # user_image_1 = request.files['user_image_1']
#     # user_image_2 = request.files['user_image_2']
#     if user_image.filename == '':
#         return {"detail": "Invalid file or filename missing"}, 400
#     user_id = request.form.get('user_id')
#     image_copy_name = 'simulated_image-{}.jpg'.format(str(user_id))
#     user_image.save(os.path.join(SIMULATOR_INPUT, image_copy_name))
#     user_image = skimage_io.imread(os.path.join(SIMULATOR_INPUT, image_copy_name))

#     r_value = request.form.get('r_value')
#     g_value = request.form.get('g_value')
#     b_value = request.form.get('b_value')
#     eyeshadow_makeup = Eyeshadow()

#     predict_result_medium = eyeshadow_makeup.apply_eyeshadow(
#         user_image, r_value, g_value, b_value, 1)
#     predict_result_fade = eyeshadow_makeup.apply_eyeshadow(
#         user_image, r_value, g_value, b_value, 1.2)
#     predict_result_intense = eyeshadow_makeup.apply_eyeshadow(
#         user_image, r_value, g_value, b_value, 1.3)

#     result = [predict_result_intense,
#               predict_result_medium, predict_result_fade]
#     encoded_img = []
#     for image_path in result:
#         encoded_img.append(get_response_image(
#             '{}/{}'.format(SIMULATOR_OUTPUT, image_path)))

#     return JSONEncoder().encode(encoded_img), 200


# @simulation.route('/api/simulator/blush', methods=['POST'])
# @cross_origin()
# def simulator_value():
#     # check if the post request has the file part
#     if 'user_image' not in request.files:
#         return {"detail": "No file found"}, 400
#     user_image = request.files['user_image']
#     # user_image_1 = request.files['user_image_1']
#     # user_image_2 = request.files['user_image_2']
#     if user_image.filename == '':
#         return {"detail": "Invalid file or filename missing"}, 400
#     user_id = request.form.get('user_id')
#     image_copy_name = 'simulated_image-{}.jpg'.format(str(user_id))
#     user_image.save(os.path.join(SIMULATOR_INPUT, image_copy_name))
#     r_value = request.form.get('r_value')
#     g_value = request.form.get('g_value')
#     b_value = request.form.get('b_value')
#     apply_makeup = ApplyMakeup()

#     predict_result_fade = apply_makeup.apply_blush(
#         image_copy_name, r_value, g_value, b_value, 121, 121, 0.1)
#     predict_result_medium = apply_makeup.apply_blush(
#         image_copy_name, r_value, g_value, b_value, 81, 81, 0.15)
#     predict_result_intense = apply_makeup.apply_blush(
#         image_copy_name, r_value, g_value, b_value, 41, 41, 0.15)

#     result = [predict_result_intense,
#               predict_result_medium, predict_result_fade]
#     encoded_img = []
#     for image_path in result:
#         encoded_img.append(get_response_image(
#             '{}/{}'.format(SIMULATOR_OUTPUT, image_path)))

#     return JSONEncoder().encode(encoded_img), 200
#     # return send_from_directory(
#     #     SIMULATOR_OUTPUT,
#     #     predict_result_medium,
#     #     mimetype='image/jpeg')


# @simulation.route('/api/simulator/foundation', methods=['POST'])
# @cross_origin()
# def foundation_value():
#     if 'user_image' not in request.files:
#         return {"detail": "No file found"}, 400
#     user_image = request.files['user_image']
#     # user_image_1 = request.files['user_image_1']
#     # user_image_2 = request.files['user_image_2']
#     if user_image.filename == '':
#         return {"detail": "Invalid file or filename missing"}, 400
#     user_id = request.form.get('user_id')
#     image_copy_name = 'simulated_image-{}.jpg'.format(str(user_id))
#     user_image.save(os.path.join(SIMULATOR_INPUT, image_copy_name))
#     r_value = request.form.get('r_value')
#     g_value = request.form.get('g_value')
#     b_value = request.form.get('b_value')
#     apply_makeup = ApplyMakeup()

#     predict_result_fade = apply_makeup.apply_foundation(
#         image_copy_name, r_value, g_value, b_value, 121, 121, 0.1)
#     # print(r_value, g_value, b_value)
#     predict_result_medium = apply_makeup.apply_foundation(
#         image_copy_name, r_value, g_value, b_value, 77, 77, 0.5)
#     predict_result_intense = apply_makeup.apply_foundation(
#         image_copy_name, r_value, g_value, b_value, 75, 75, 1.1)

#     result = [predict_result_intense,
#               predict_result_medium, predict_result_fade]
#     encoded_img = []
#     for image_path in result:
#         encoded_img.append(get_response_image(
#             '{}/{}'.format(SIMULATOR_OUTPUT, image_path)))

#     return JSONEncoder().encode(encoded_img), 200
#     return send_from_directory(
#         SIMULATOR_OUTPUT,
#         predict_result_medium,
#         mimetype='image/jpeg')


# # -----------------------------------------------------------------


@simulation.route('/api/opencam', methods=['GET'])
@cross_origin()
def opencam():
    mutils.start_cam()
    return "Success opening cam", 200


@simulation.route('/api/closecam', methods=['GET'])
@cross_origin()
def close_cam():
    mutils.stop_cam()
    return 'Cam closed'


@simulation.route('/api/video_feed', methods=['GET'])
@cross_origin()
def video_feed():
    return Response(mutils.apply_makeup(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@simulation.route('/api/video/<makeup_type>', methods=['POST'])
@cross_origin()
def video_eyeshadow(makeup_type):
    print(makeup_type)
    req_data = request.get_json(force=True)
    input_args = [
        makeup_type,
        req_data.get('r_value'),
        req_data.get('g_value'),
        req_data.get('b_value'),
        req_data.get('intensity'),
        # req_data.get('l_type'),
        # req_data.get('gloss'),
        # req_data.get('k_h'),
        # req_data.get('k_w')
    ]
    print(input_args)
    makeup_args = {x: y for x, y in zip(mutils.Globals.makeup_args, input_args) if y is not None}
    print(makeup_args)
    mutils.enable_makeup(**makeup_args)
    return (makeup_type, 200)

@simulation.route('/api/video_off/<makeup_type>', methods=['GET'])
@cross_origin()
def video_no_eyeshadow(makeup_type):
    mutils.disable_makeup(makeup_type)
    print(makeup_type, "off")
    return ("off", 200)



@simulation.route('/api/test/blush', methods=['GET'])
def blush():
    mutils.handle_makeup_state('blush', 130, 197, 81, .6)


@simulation.route('/api/test/eye', methods=['GET'])
def eye():
    mutils.handle_makeup_state('eyeshadow', 237, 29, 36, .3)


# This method executes after every API request.
@simulation.after_request
def after_request(response):
    return response
