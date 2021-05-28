# Import necessary libraries
import threading

from flask import Flask, json, render_template, Response, request
import cv2
import atexit
import src.cv.makeup.utils as mutils
import atexit

# Initialize the Flask app
app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/open-cam', methods=['POST'])
def open_cam():
    mutils.start_cam()
    return "Success opening cam", 200


@app.route('/close-cam', methods=['POST'])
def close_cam():
    mutils.stop_cam()
    return 'Cam closed'


@app.route('/video-feed')
def video_feed():
    return Response(mutils.apply_makeup(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/blush')
def blush():
    mutils.enable_makeup('blush', 130, 197, 81, .6)
    return 'ok'


@app.route('/eyeshadow')
def eyeshadow():
    mutils.enable_makeup('eyeshadow', 130, 197, 81, .6)
    return 'ok'

@app.route('/lipstick')
def lipstick():
    mutils.enable_makeup('lipstick', 200, 10, 20)
    return 'ok'


def clean_exit():
    print('Making sure camera is turned off to exit properly.')
    mutils.stop_cam()


@app.route('/enable/<makeup_type>', methods=['POST'])
def enabel_makeup(makeup_type):
    print(makeup_type)
    req_data = request.get_json()
    input_args = [
        makeup_type,
        req_data.get('r_value'),
        req_data.get('g_value'),
        req_data.get('b_value'),
        req_data.get('intensity')
    ]
    print(input_args)
    makeup_args = {x: y for x, y in zip(mutils.Globals.makeup_args, input_args) if y is not None}
    print(makeup_args)
    mutils.enable_makeup(**makeup_args)
    return makeup_type, 200


@app.route('/disable/<makeup_type>', methods=['POST'])
def disable_makeup(makeup_type):
    mutils.disable_makeup(makeup_type)
    return f'{makeup_type} deactivated' , 200


atexit.register(clean_exit)


if __name__ == '__main__':
    # mutils.Globals.cap = cv2.VideoCapture(1)
    app.run(debug=True)
