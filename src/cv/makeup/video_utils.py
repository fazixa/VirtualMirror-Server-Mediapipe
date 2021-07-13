import cv2
import mediapipe as mp
from numpy import ndarray
from skimage.draw import polygon
from mediapipe.python.solutions import face_detection
from mediapipe.python.solutions import face_mesh
import time
from src.cv.makeup import commons, constants
# from src.cv.simulation.apply_lens import Lens

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


class Makeup_Worker:
    def __init__(self, bounds=None, instance=None) -> None:
        self.bounds = bounds
        self.instance = instance() if instance is not None else None


class Globals:
    cap = cv2.VideoCapture()
    face_detector = face_detection.FaceDetection(min_detection_confidence=.5)
    face_mesher = face_mesh.FaceMesh(min_detection_confidence=.5, min_tracking_confidence=.5)
    makeup_workers = {}

    blush       = Makeup_Worker()
    eyeliner    = Makeup_Worker()
    lipstick    = Makeup_Worker()
    concealer   = Makeup_Worker()
    eyeshadow   = Makeup_Worker()
    foundation  = Makeup_Worker()
    # lens        = Makeup_Worker(instance=Lens)

    #### Motion Detection Vars ####
    prev_frame = None
    motion_detected = True

    f_xmin = None
    f_ymin = None
    f_width = None
    f_height = None
    ################################

    makeup_args = None

    idx_to_coordinates = []
    output_frame = None
    # face_resize_width = 500


def lipstick_worker_video(image, r, g, b, intensity, gloss) -> None:
    for region in constants.LIPS:
        roi_x = []
        roi_y = []
        for point in region:
            roi_x.append(Globals.idx_to_coordinates[point][0])
            roi_y.append(Globals.idx_to_coordinates[point][1])

        margin = 4
        top_x = min(roi_x)-margin
        top_y = min(roi_y)-margin
        bottom_x = max(roi_x)+margin
        bottom_y = max(roi_y)+margin

        rr, cc = polygon(roi_x, roi_y)
        
        crop = image[top_y:bottom_y, top_x:bottom_x,]
        if gloss:
            crop = commons.moist(crop, cc-top_y,rr-top_x, 220)
        crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, b, g, r, intensity)
        crop = commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 21, 7)
        image[top_y:bottom_y, top_x:bottom_x,] = crop

    return image


def eyeshadow_worker_video(image, r, g, b, intensity) -> None:
    for region in constants.EYESHADOW:
        roi_x = []
        roi_y = []
        for point in region:
            roi_x.append(Globals.idx_to_coordinates[point][0])
            roi_y.append(Globals.idx_to_coordinates[point][1])

        margin = 10
        top_x = min(roi_x)-margin
        top_y = min(roi_y)-margin
        bottom_x = max(roi_x)+margin
        bottom_y = max(roi_y)+margin

        rr, cc = polygon(roi_x, roi_y)
        
        crop = image[top_y:bottom_y, top_x:bottom_x,]
        crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, b, g, r, intensity)
        crop = commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 31, 15)

        image[top_y:bottom_y, top_x:bottom_x,] = crop
    
    return image


Globals.makeup_workers = {
    # 'foundation_worker':    { 'function': foundation_worker,    'static_function': foundation_worker_static, 'args': [], 'enabled': False, 'enabled_first': False },
    # 'concealer_worker':     { 'function': concealer_worker,     'static_function': concealer_worker_static,  'args': [], 'enabled': False },
    # 'blush_worker':         { 'function': blush_worker,         'static_function': blush_worker_static,      'args': [], 'enabled': False },
    'eyeshadow_worker':     { 'function': eyeshadow_worker_video,     'static_function': eyeshadow_worker_video, 'args': [], 'enabled': False },
    # 'eyeliner_worker':      { 'function': eyeliner_worker,      'static_function': eyeliner_worker_static,   'args': [], 'enabled': False },
    'lipstick_worker':      { 'function': lipstick_worker_video,      'static_function': lipstick_worker_video,  'args': [], 'enabled': False },
    # 'lens_worker':          { 'function': lens_worker,          'static_function': lens_worker_static,       'args': [], 'enabled': False, 'enabled_first': False},
}


def join_makeup_workers_video(image) -> ndarray:
    result: ndarray = image

    for makeup_worker in Globals.makeup_workers:
        worker = Globals.makeup_workers[makeup_worker]

        if worker['enabled']:
            result = worker['function'](result, *worker['args'])

    return result


def apply_makeup_video(filepath) -> None:
    cap = cv2.VideoCapture('video.mp4')

    video = cv2.VideoWriter('out.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            cap.get(cv2.CAP_PROP_FPS),
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    result = []

    if not cap.isOpened(): 
        print("Error reading video file")
    else:
        start = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = Globals.face_mesher.process(frame)
            # image.flags.writeable = True

            if results.multi_face_landmarks:
                for landmark_list in results.multi_face_landmarks:
        
                    image_rows, image_cols, _ = frame.shape
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

                Globals.idx_to_coordinates = idx_to_coordinates

            result = join_makeup_workers_video(frame)

            video.write(result)

        print(time.time() - start)

    cap.release()
    video.release()


def enable_makeup(makeup_type='', r=0, g=0, b=0, intensity=.7, gloss=False):
    """Enables the requested makeup so that the joining methods will be applied it in the next frame.
    It also passes the received arguments from the API to the corresponding function.

    Args:
        arg1 (string): name of the makeup, it can be one of:
            [eyeshadow, lipstick, blush, concealer, foundation, eyeliner, lens]
        arg2 (int): rgb value of red color 
        arg3 (int): rgb value of green color 
        arg4 (int): rgb value of blue color
        arg5 (float): intensity of the applied makeup
        arg6 (bool): Determines whethere gloss needs to be applied or not (only for lipstick)
    """

    Globals.motion_detected = True
    
    if makeup_type == 'eyeshadow':
        Globals.makeup_workers['eyeshadow_worker']['args'] = [r, g, b, intensity]
        Globals.makeup_workers['eyeshadow_worker']['enabled'] = True
    elif makeup_type == 'lipstick':
        Globals.makeup_workers['lipstick_worker']['args'] = [r, g, b, intensity, gloss]
        Globals.makeup_workers['lipstick_worker']['enabled'] = True
    elif makeup_type == 'blush':
        Globals.makeup_workers['blush_worker']['args'] = [r, g, b, intensity]
        Globals.makeup_workers['blush_worker']['enabled'] = True
    elif makeup_type == 'concealer':
        Globals.makeup_workers['concealer_worker']['args'] = [r, g, b, intensity]
        Globals.makeup_workers['concealer_worker']['enabled'] = True
    elif makeup_type == 'foundation':
        Globals.makeup_workers['foundation_worker']['args'] = [r, g, b, intensity]
        Globals.makeup_workers['foundation_worker']['enabled_first'] = True
    elif makeup_type == 'eyeliner':
        Globals.makeup_workers['eyeliner_worker']['args'] = [r, g, b, intensity]
        Globals.makeup_workers['eyeliner_worker']['enabled'] = True
    elif makeup_type == 'lens':
        Globals.makeup_workers['lens_worker']['args'] = [r, g, b, intensity]
        Globals.makeup_workers['lens_worker']['enabled_first'] = True


Globals.makeup_args = enable_makeup.__code__.co_varnames


def disable_makeup(makeup_type):
    """Disables the requested makeup so that the joining methods will not be applied it in the next frame.

    Args:
        arg1 (string): name of the makeup, it can be one of:
            [eyeshadow, lipstick, blush, concealer, foundation, eyeliner, lens]
            
    """
    
    Globals.motion_detected = True

    if makeup_type == 'eyeshadow':
        Globals.makeup_workers['eyeshadow_worker']['enabled'] = False
    elif makeup_type == 'lipstick':
        Globals.makeup_workers['lipstick_worker']['enabled'] = False
    elif makeup_type == 'blush':
        Globals.makeup_workers['blush_worker']['enabled'] = False
    elif makeup_type == 'concealer':
        Globals.makeup_workers['concealer_worker']['enabled'] = False
    elif makeup_type == 'foundation':
        Globals.makeup_workers['foundation_worker']['enabled'] = False
    elif makeup_type == 'eyeliner':
        Globals.makeup_workers['eyeliner_worker']['enabled'] = False
    elif makeup_type == 'lens':
        Globals.makeup_workers['lens_worker']['enabled_first'] = False