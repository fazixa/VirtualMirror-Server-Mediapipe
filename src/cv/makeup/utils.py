import cv2, threading, mediapipe as mp
from numpy import ndarray
from skimage.draw import polygon
from mediapipe.python.solutions import face_detection
from mediapipe.python.solutions import face_mesh
import traceback

from src.cv.makeup import commons, constants
from src.cv.simulation.apply_lens import Lens

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


class Makeup_Worker:
    """Defines a makeup worker instance

    Args:
        arg1 (list): list of the bounds related to the makeup worker, it's assigned in normal worker functions and used in static worker functions
        arg2 (class): an instance of the related makeup worker, in case it exists (E.g. Lens)

    """
    def __init__(self, bounds=None, instance=None) -> None:
        self.bounds = bounds
        self.instance = instance() if instance is not None else None


class Globals:
    """Holds global values that are used by static workers and other functions

    Properties:
        :cap: Video captured from webcam, can be closed and open upon request
        :face_detector: Instance of MediaPipe face_detection module
        :face_mesher: Instance of MediaPipe face_mesh module
        :makeup_workers: Dictionary of the makeup workers and their related information
        :makeup args: parameter names that are given to makeup workers, used in the api designed for this purpose
        :makeup instances:
            - lens
            - eyeliner
            - eyeshadow
            - blush
            - lipstick
            - foundation
            - concealer
        :motion detection variables:
            - **prev_frame**: holds the previos frame for comparing with the current frame
            - **motion detected** (bool): True if motion detected, False otherwise, starts with True
            - **f_xmin**: minimum x value among points of face crop
            - **f_ymin**: minimum y value among points of face crop
            - **f_width**: width of the face crop
            - **f_height**: height of the face crop
        :idx_to_coordinates: holds to coordinates calculated by turning face mesh index points to real coordinates in given image of face
        :output_frame: holds the given image without any change for using in static function or in special conditions
    """

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
    lens        = Makeup_Worker(instance=Lens)

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



############################################################## WORKERS #################################################################

def concealer_worker(image, r, g, b, intensity, out_queue) -> None:
    """This function applies a concealer effect on an input image.
    This function is called by a threading function and its output
    is appended to a given list to be processed later.

    Args:
        arg1 (ndarray)  : input image, a crop of the face found in camera viewport
        arg2 (int)      : rgb value of red color 
        arg3 (int)      : rgb value of green color 
        arg4 (int)      : rgb value of blue color
        arg5 (float)    : intensity of the applied makeup
        arg6 (list)     : shared list for appending the output
        
    """

    crops = []
    bounds = []

    for region in constants.CONCEALER:
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

        crops.append(commons.apply_blur_color(crop, cc-top_y, rr-top_x, b, g, r, 0.5))
        bounds.append([rr, cc, top_x, top_y, bottom_x, bottom_y])

    # Globals.concealer.crops = crops
    Globals.concealer.bounds = bounds

    out_queue.append({
        'index': 4,
        'crops': crops,
        'bounds': bounds
    })


def concealer_worker_static(image, r, g, b, intensity, out_queue) -> None:
    """This function applies a concealer effect on an input image
    with the calculated boundaries in the normal worker function.
    This function is called by a threading function and its output
    is appended to a given list to be processed later.

    Args:
        arg1 (ndarray)  : input image, a crop of the face found in camera viewport
        arg2 (int)      : rgb value of red color 
        arg3 (int)      : rgb value of green color 
        arg4 (int)      : rgb value of blue color
        arg5 (float)    : intensity of the applied makeup
        arg6 (list)     : shared list for appending the output
        
    """
    crops = []

    for [rr, cc, top_x, top_y, bottom_x, bottom_y] in Globals.concealer.bounds:
        crop = image[top_y:bottom_y, top_x:bottom_x]
        crops.append(commons.apply_blur_color(crop, cc-top_y, rr-top_x, b, g, r, 0.5))

    out_queue.append({
        'index': 4,
        'crops': crops,
        'bounds': Globals.concealer.bounds
    })


def blush_worker(image, r, g, b, intensity, out_queue) -> None:
    """This function applies a blush effect on an input image.
    This function is called by a threading function and its output
    is appended to a given list to be processed later.

    Args:
        arg1 (ndarray)  : input image, a crop of the face found in camera viewport
        arg2 (int)      : rgb value of red color 
        arg3 (int)      : rgb value of green color 
        arg4 (int)      : rgb value of blue color
        arg5 (float)    : intensity of the applied makeup
        arg6 (list)     : shared list for appending the output
        
    """
    
    crops = []
    bounds = []

    for region in constants.BLUSH:
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
        
        crop = image[top_y:bottom_y, top_x:bottom_x, ]

        crops.append(commons.apply_blur_color(crop, cc-top_y, rr-top_x, b, g, r, 0.5))
        bounds.append([rr, cc, top_x, top_y, bottom_x, bottom_y])

    # Globals.blush.crops = crops
    Globals.blush.bounds = bounds

    out_queue.append({
        'index': 3,
        'crops': crops,
        'bounds': bounds
    })


def blush_worker_static(image, r, g, b, intensity, out_queue) -> None:
    """This function applies a blush effect on an input image
    with the calculated boundaries in the normal worker function.
    This function is called by a threading function and its output
    is appended to a given list to be processed later.

    Args:
        arg1 (ndarray)  : input image, a crop of the face found in camera viewport
        arg2 (int)      : rgb value of red color 
        arg3 (int)      : rgb value of green color 
        arg4 (int)      : rgb value of blue color
        arg5 (float)    : intensity of the applied makeup
        arg6 (list)     : shared list for appending the output
        
    """

    crops = []

    for [rr, cc, top_x, top_y, bottom_x, bottom_y] in Globals.blush.bounds:
        crop = image[top_y:bottom_y, top_x:bottom_x]
        crops.append(commons.apply_blur_color(crop, cc-top_y, rr-top_x, b, g, r, 0.5))

    out_queue.append({
        'index': 3,
        'crops': crops,
        'bounds': Globals.blush.bounds
    })


def lipstick_worker(image, r, g, b, intensity, gloss, out_queue) -> None:
    """This function applies a lipstick effect on an input image.
    This function is called by a threading function and its output
    is appended to a given list to be processed later.

    Args:
        arg1 (ndarray)  : input image, a crop of the face found in camera viewport
        arg2 (int)      : rgb value of red color 
        arg3 (int)      : rgb value of green color 
        arg4 (int)      : rgb value of blue color
        arg5 (float)    : intensity of the applied makeup
        arg6 (list)     : shared list for appending the output
        
    """

    crops = []
    bounds = []

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
        crops.append(commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 21, 7))
        bounds.append([rr, cc, top_x, top_y, bottom_x, bottom_y])

    # Globals.lipstick.crops = crops
    Globals.lipstick.bounds = bounds

    out_queue.append({
        'index': 0,
        'crops': crops,
        'bounds': bounds
    })


def lisptick_worker_static(image, r, g, b, intensity, gloss, out_queue) -> None:
    crops = []

    for [rr, cc, top_x, top_y, bottom_x, bottom_y] in Globals.lipstick.bounds:
        crop = image[top_y:bottom_y, top_x:bottom_x,]
        if gloss:
           crop = commons.moist(crop, cc-top_y,rr-top_x, 220)
        crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, b, g, r, intensity)
        crops.append(commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 21, 7))

    out_queue.append({
        'index': 0,
        'crops': crops,
        'bounds': Globals.lipstick.bounds
    })


def eyeshadow_worker(image, r, g, b, intensity, out_queue) -> None:
    """This function applies an eyeshadow effect on an input image.
    This function is called by a threading function and its output
    is appended to a given list to be processed later.

    Args:
        arg1 (ndarray)  : input image, a crop of the face found in camera viewport
        arg2 (int)      : rgb value of red color 
        arg3 (int)      : rgb value of green color 
        arg4 (int)      : rgb value of blue color
        arg5 (float)    : intensity of the applied makeup
        arg6 (list)     : shared list for appending the output
        
    """

    crops = []
    bounds = []

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
        crops.append(commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 31, 15))
        bounds.append([rr, cc, top_x, top_y, bottom_x, bottom_y])

    # Globals.eyeshadow.crops = crops
    Globals.eyeshadow.bounds = bounds

    out_queue.append({
        'index': 1,
        'crops': crops,
        'bounds': bounds
    })


def eyeshadow_worker_static(image, r, g, b, intensity, out_queue) -> None:
    crops = []

    for [rr, cc, top_x, top_y, bottom_x, bottom_y] in Globals.eyeshadow.bounds:
        crop = image[top_y:bottom_y, top_x:bottom_x,]
        crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, b, g, r, intensity)
        crops.append(commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 31, 15))

    out_queue.append({
        'index': 1,
        'crops': crops,
        'bounds': Globals.eyeshadow.bounds
    })


def eyeliner_worker(image, r, g, b, intensity, out_queue) -> None:
    """This function applies an eyeliner effect on an input image.
    This function is called by a threading function and its output
    is appended to a given list to be processed later.

    Args:
        arg1 (ndarray)  : input image, a crop of the face found in camera viewport
        arg2 (int)      : rgb value of red color 
        arg3 (int)      : rgb value of green color 
        arg4 (int)      : rgb value of blue color
        arg5 (float)    : intensity of the applied makeup
        arg6 (list)     : shared list for appending the output
        
    """

    crops = []
    bounds = []

    for region in constants.EYELINER:
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
        crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, b, g, r, 1.5)
        crops.append(commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 15, 5))
        bounds.append([rr, cc, top_x, top_y, bottom_x, bottom_y])

    # Globals.eyeliner.crops = crops
    Globals.eyeliner.bounds = bounds

    out_queue.append({
        'index': 2,
        'crops': crops,
        'bounds': bounds
    })


def eyeliner_worker_static(image, r, g, b, intensity, out_queue) -> None:
    crops = []

    for [rr, cc, top_x, top_y, bottom_x, bottom_y] in Globals.eyeliner.bounds:
        crop = image[top_y:bottom_y, top_x:bottom_x,]
        crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, b, g, r, 1.5)
        crops.append(commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 15, 5))

    out_queue.append({
        'index': 2,
        'crops': crops,
        'bounds': Globals.eyeliner.bounds
    })


def foundation_worker(image, r, g, b, intensity) -> ndarray:
    """This function applies a foundation effect on an input image.
    This function is called by a threading function and its output
    is appended to a given list to be processed later.

    Args:
        arg1 (ndarray)  : input image, a crop of the face found in camera viewport
        arg2 (int)      : rgb value of red color 
        arg3 (int)      : rgb value of green color 
        arg4 (int)      : rgb value of blue color
        arg5 (float)    : intensity of the applied makeup
        
    Returns:
        ndarray: The given image with the foundation effect applied on it, this image will be
        used as an input to other activated makeup workers in order for them to apply their effect
        on this image.
    """

    bounds = []

    for region in constants.FOUNDATION:
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
        # crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, b, g, r, intensity)
        # image[top_y:bottom_y, top_x:bottom_x,] = commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 15, 5)
        image[top_y:bottom_y, top_x:bottom_x,] = commons.apply_blur_color(crop, cc-top_y ,rr-top_x, b, g, r, intensity)

        bounds.append([rr, cc, top_x, top_y, bottom_x, bottom_y])

    Globals.foundation.bounds = bounds

    return image

def foundation_worker_static(image, r, g, b, intensity) -> ndarray:    
    for [rr, cc, top_x, top_y, bottom_x, bottom_y] in Globals.foundation.bounds:
        crop = image[top_y:bottom_y, top_x:bottom_x,]
        # crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, b, g, r, intensity)
        # image[top_y:bottom_y, top_x:bottom_x,] = commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 15, 5)
        image[top_y:bottom_y, top_x:bottom_x,] = commons.apply_blur_color(crop, cc-top_y ,rr-top_x, b, g, r, intensity)
    
    return image


def lens_worker(image, r, g, b, intensity) -> ndarray:
    """This function applies a contact lens effect on an input image.
    This function is called by a threading function and its output
    is appended to a given list to be processed later.

    Args:
        arg1 (ndarray)  : input image, a crop of the face found in camera viewport
        arg2 (int)      : rgb value of red color 
        arg3 (int)      : rgb value of green color 
        arg4 (int)      : rgb value of blue color
        arg5 (float)    : intensity of the applied makeup
        
    Returns:
        ndarray: The given image with the contact effect applied on it, this image will be
        used as an input to other activated makeup workers in order for them to apply their effect
        on this image.
    """

    return Globals.lens.instance.apply_lens(image, r, g, b)

def lens_worker_static(image, r, g, b, intensity) -> ndarray:    
    return Globals.lens.instance.apply_lens_static(image, r, g, b)


####################################################################################################################################


Globals.makeup_workers = {
    'lens_worker':          { 'function': lens_worker,          'static_function': lens_worker_static,       'args': [], 'enabled': False, 'enabled_first': False},
    'lipstick_worker':      { 'function': lipstick_worker,      'static_function': lisptick_worker_static,   'args': [], 'enabled': False },
    'eyeshadow_worker':     { 'function': eyeshadow_worker,     'static_function': eyeshadow_worker_static,  'args': [], 'enabled': False },
    'eyeliner_worker':      { 'function': eyeliner_worker,      'static_function': eyeliner_worker_static,   'args': [], 'enabled': False },
    'blush_worker':         { 'function': blush_worker,         'static_function': blush_worker_static,      'args': [], 'enabled': False },
    'concealer_worker':     { 'function': concealer_worker,     'static_function': concealer_worker_static,  'args': [], 'enabled': False },
    'foundation_worker':    { 'function': foundation_worker,    'static_function': foundation_worker_static, 'args': [], 'enabled': False, 'enabled_first': False },
}


def join_makeup_workers(image):
    threads = []
    shared_list = []

    if Globals.makeup_workers['foundation_worker']['enabled_first']:
        image = foundation_worker(image, *Globals.makeup_workers['foundation_worker']['args'])

    if Globals.makeup_workers['lens_worker']['enabled_first']:
        image = lens_worker(image, *Globals.makeup_workers['lens_worker']['args'])

    for makeup_worker in Globals.makeup_workers:
        worker = Globals.makeup_workers[makeup_worker]

        if worker['enabled']:
            t = threading.Thread(
                target=worker['function'],
                args=(image, *worker['args'], shared_list),
                daemon=True
            )
            threads.append(t)

    if len(threads) > 0:
        for t in threads:
            t.start()
            t.join()

        shared_list = sorted(shared_list, key=lambda x: x['index'], reverse=True)

        while len(shared_list) > 0:
            temp_img = shared_list.pop()
            
            for crop, [rr, cc, top_x, top_y, _, _] in zip(temp_img['crops'], temp_img['bounds']):
                image[cc, rr] = crop[cc-top_y, rr-top_x]

    return image


def join_makeup_workers_static(image):
    threads = []
    shared_list = []

    if Globals.makeup_workers['foundation_worker']['enabled_first']:
        image = foundation_worker_static(image, *Globals.makeup_workers['foundation_worker']['args'])

    if Globals.makeup_workers['lens_worker']['enabled_first']:
        image = lens_worker_static(image, *Globals.makeup_workers['lens_worker']['args'])

    for makeup_worker in Globals.makeup_workers:
        worker = Globals.makeup_workers[makeup_worker]

        if worker['enabled']:
            t = threading.Thread(
                target=worker['static_function'],
                args=(image, *worker['args'], shared_list),
                daemon=True
            )
            threads.append(t)

    if len(threads) > 0:
        for t in threads:
            t.start()
            t.join()

        shared_list = sorted(shared_list, key=lambda x: x['index'], reverse=True)

        while len(shared_list) > 0:
            temp_img = shared_list.pop()
            
            for crop, [rr, cc, top_x, top_y, _, _] in zip(temp_img['crops'], temp_img['bounds']):
                image[cc, rr] = crop[cc-top_y, rr-top_x]

    return image


def apply_makeup():
        while Globals.cap.isOpened():
            success, image = Globals.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue


            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False

            Globals.output_frame = image

            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if Globals.prev_frame is None:
                Globals.prev_frame = gray
                continue
    
            frame_diff = cv2.absdiff(Globals.prev_frame, gray)

            frame_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1] 
            frame_thresh = cv2.dilate(frame_thresh, None, iterations=2)

            cnts, _ = cv2.findContours(frame_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

            for contour in cnts: 
                temp = cv2.contourArea(contour)
                if temp < 300:  
                    continue
                Globals.motion_detected = True

            
            if Globals.motion_detected:
                print('motion detected')

                results = Globals.face_detector.process(image)

                im_height, im_width = image.shape[:2]

                if results.detections:
                    for detection in results.detections:
                        Globals.f_xmin = f_xmin = int((detection.location_data.relative_bounding_box.xmin - .1) * im_width)
                        Globals.f_ymin = f_ymin = int((detection.location_data.relative_bounding_box.ymin - .2) * im_height)
                        Globals.f_width = f_width = int((detection.location_data.relative_bounding_box.width + .2) * im_width)
                        Globals.f_height = f_height = int((detection.location_data.relative_bounding_box.height + .25) * im_height)

                    if f_xmin < 0 or f_ymin < 0:
                        print('Face outside of camera view, please move inside')
                        continue

                    face_crop = image[f_ymin:f_ymin+f_height, f_xmin:f_xmin+f_width]

                    # crop_height, crop_width = face_crop.shape[:2]

                    # face_crop = cv2.resize(face_crop,
                    #             (Globals.face_resize_width, int(crop_height * Globals.face_resize_width / float(crop_width))),
                    #             interpolation=cv2.INTER_AREA)

                    # image.flags.writeable = False
                    results = Globals.face_mesher.process(face_crop)
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

                        Globals.idx_to_coordinates = idx_to_coordinates
            

                        try:
                            face_crop = join_makeup_workers(face_crop)
                            # face_crop = cv2.resize(face_crop, (crop_width, crop_height), interpolation=cv2.INTER_AREA)
                            image[f_ymin:f_ymin+f_height, f_xmin:f_xmin+f_width] = face_crop
                        except Exception as e:
                            print(e)

                        
                        Globals.motion_detected = False

            else:
                print('no motion')
                face_crop = image[Globals.f_ymin:Globals.f_ymin+Globals.f_height, Globals.f_xmin:Globals.f_xmin+Globals.f_width]
                
                # crop_height, crop_width = face_crop.shape[:2]
                # face_crop = cv2.resize(face_crop,
                #                 (Globals.face_resize_width, int(crop_height * Globals.face_resize_width / float(crop_width))),
                #                 interpolation=cv2.INTER_AREA)

                try:
                    face_crop = join_makeup_workers_static(face_crop)
                    # face_crop = cv2.resize(face_crop, (crop_width, crop_height), interpolation=cv2.INTER_AREA)
                    image[Globals.f_ymin:Globals.f_ymin+Globals.f_height, Globals.f_xmin:Globals.f_xmin+Globals.f_width] = face_crop
                except Exception as e:
                    traceback.print_exc()


            Globals.prev_frame = gray.copy()

            # return image

            (flag, encodedImage) = cv2.imencode(".jpg", image)
            
            # ensure the frame was successfully encoded
            if not flag:
                continue
            # yield the output frame in the byte format
            yield (b'--frame\r\n' b'Content-Type: image/jpg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')


def enable_makeup(makeup_type='', r=0, g=0, b=0, intensity=.7, gloss=False):
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


def start_cam():
    Globals.cap.open(2)

def stop_cam():
    Globals.cap.release()