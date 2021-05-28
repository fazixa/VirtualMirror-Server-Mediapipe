# settings.py
import os
# __file__ refers to the file settings.py
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top

# User path
USER_IMAGE_PATH =os.path.join(APP_ROOT,  './image/user_image')
USER_IMAGE_FILE_TYPE = '.png'

# Model path
SHAPE_68_PATH = os.path.join(
    APP_ROOT, './cv/train_model/shape_predictor_68_face_landmarks.dat')
SHAPE_81_PATH = os.path.join(APP_ROOT, './cv/train_model/shape_predictor_81_face_landmarks.dat')
SKIN_CLUSTER_MODEL_PATH = os.path.join(APP_ROOT,'./cv/train_model/skin_tone_model.sav')


# Color prediction settings
COLOR_PREDICTION_INPUT = os.path.join(
    APP_ROOT, './image/color_prediction_input')
COLOR_PREDICTION_OUTPUT = os.path.join(
    APP_ROOT, './image/color_prediction_output')
METHOD_NUM = 2
BLACK_THRESHOLD = 20
COLOR_COMPARE_VAL = 10
RETURN_SIZE = 50
SAVE_FILE_TYPE = '.jpg'


# Simulation
SIMULATOR_INPUT = os.path.join(APP_ROOT, 'image/simulator_input')
SIMULATOR_OUTPUT = os.path.join(APP_ROOT, 'image/simulator_output')


# Kmean settings
NUMBER_K = 5
RETURN_INDEX = 2
