import io
from skimage import io as skimage_io
import os
from flask import Flask, request, send_from_directory, jsonify, Blueprint
from flask_cors import cross_origin
from PIL import Image
from base64 import encodebytes
from src.route.json_encode import JSONEncoder
# from src.cv.simulation.apply_eyeliner import Eyeliner
from src.cv.makeup import commons, constants
from mediapipe.python.solutions import face_detection
from mediapipe.python.solutions import face_mesh
from skimage.draw import polygon

from src.settings import SIMULATOR_INPUT, SIMULATOR_OUTPUT
import cv2
import time
import imutils
from flask import Flask, render_template, url_for, request, Response
import dlib
from src.settings import SHAPE_68_PATH

eyelinerm = Blueprint('eyeliner', __name__)


# This method executes before any API request


@eyelinerm.before_request
def before_request():
    print('Start eyeliner API request')

# --------------------------------- generl defs

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r')  # reads the PIL image
    byte_arr = io.BytesIO()
    # convert the PIL image to byte array
    pil_img.save(byte_arr, format='JPEG')
    encoded_img = encodebytes(byte_arr.getvalue()).decode(
        'ascii')  # encode as base64
    return encoded_img

# ------------------------------------ non general
@eyelinerm.route('/api/makeup/image/eyeliner', methods=['POST'])
@cross_origin()
def simulator_lip():
    print("eyeeeee")
    # check if the post request has the file part
    if 'user_image' not in request.files:
        return {"detail": "No file found"}, 400
    user_image = request.files['user_image']
    if user_image.filename == '':
        return {"detail": "Invalid file or filename missing"}, 400
    user_id = request.form.get('user_id')
    image_copy_name = 'simulated_image-{}.jpg'.format(str(user_id))
    user_image.save(os.path.join(SIMULATOR_INPUT, image_copy_name))
    user_image = cv2.imread(os.path.join(SIMULATOR_INPUT, image_copy_name))
    

    r = int(request.form.get('r_value'))
    g = int(request.form.get('g_value'))
    b = int(request.form.get('b_value'))
    print(r, "hiiiiiiiiiiiiiiiiii")

    face_detector = face_detection.FaceDetection(min_detection_confidence=.5)
    face_mesher = face_mesh.FaceMesh(min_detection_confidence=.5, min_tracking_confidence=.5)

    results = face_detector.process(user_image)

    im_height, im_width = user_image.shape[:2]
    
    if results.detections:
        for detection in results.detections:
            f_xmin = int((detection.location_data.relative_bounding_box.xmin - .1) * im_width)
            f_ymin = int((detection.location_data.relative_bounding_box.ymin - .2) * im_height)
            f_width = int((detection.location_data.relative_bounding_box.width + .2) * im_width)
            f_height = int((detection.location_data.relative_bounding_box.height + .25) * im_height)

        if f_xmin < 0 or f_ymin < 0:
             return {"detail": "No face found"}, 400


        face_crop = user_image[f_ymin:f_ymin+f_height, f_xmin:f_xmin+f_width]

        # image.flags.writeable = False
        results = face_mesher.process(face_crop)
        # image.flags.writeable = True

        if results.multi_face_landmarks:
            for landmark_list in results.multi_face_landmarks:
    
                image_rows, image_cols, _ = face_crop.shape
                idx_to_coordinates = {}
                for idx, landmark in enumerate(landmark_list.landmark):
                    
                    if ((landmark.HasField('visibility') and
                        landmark.visibility < constants.VISIBILITY_THRESHOLD) or
                        (landmark.HasField('presence') and
                        landmark.presence < constants.PRESENCE_THRESHOLD)):
                        continue
                    landmark_px = commons._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                image_cols, image_rows)

                    if landmark_px:
                        idx_to_coordinates[idx] = landmark_px


            for region in constants.EYELINER:
                roi_x = []
                roi_y = []
                for point in region:
                    roi_x.append(idx_to_coordinates[point][0])
                    roi_y.append(idx_to_coordinates[point][1])

                margin = 10
                top_x = min(roi_x)-margin
                top_y = min(roi_y)-margin
                bottom_x = max(roi_x)+margin
                bottom_y = max(roi_y)+margin

                rr, cc = polygon(roi_x, roi_y)
                
                crop = face_crop[top_y:bottom_y, top_x:bottom_x,]
                crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, r, g, b, 0.7)
                crop_makeup = commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 15, 5)
                face_crop[top_y:bottom_y, top_x:bottom_x] = crop_makeup

                user_image[f_ymin:f_ymin+f_height, f_xmin:f_xmin+f_width] = face_crop

            predict_result_intense = save_iamge(user_image,r,g,b,"eyeshadow",0.7)
            predict_result_medium = save_iamge(user_image,r,g,b,"eyeshadow",0.5)
            predict_result_fade = save_iamge(user_image,r,g,b,"eyeshadow",0.3)





    result = [predict_result_intense,
              predict_result_medium, predict_result_fade]
    encoded_img = []
    for image_path in result:
        encoded_img.append(get_response_image(
            '{}/{}'.format(SIMULATOR_OUTPUT, image_path)))

    return (JSONEncoder().encode(encoded_img), 200)


def save_iamge(img,r_value,g_value,b_value,makeup_type,intensity):
    name = 'color_' + str(r_value) + '_' + \
    str(g_value) + '_' + str(b_value)
    file_name = '{}_output-{}_{}.jpg'.format(makeup_type,intensity, name)
    cv2.imwrite(os.path.join(SIMULATOR_OUTPUT, file_name), img)
    return file_name