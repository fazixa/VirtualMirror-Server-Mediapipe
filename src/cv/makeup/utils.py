import cv2
import threading
import mediapipe as mp
from skimage.draw import polygon
from mediapipe.python.solutions import face_detection
from mediapipe.python.solutions import face_mesh

from src.cv.makeup import commons, constants

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


class Makeup_Worker:
    def __init__(self, result=None, bounds=None) -> None:
        self.crops = result
        self.bounds = bounds


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

    #### Motion Detection Vars ####
    prev_frame = None
    motion_detected = True

    f_xmin = None
    f_ymin = None
    f_width = None
    f_height = None
    ################################

    idx_to_coordinates = []
    output_frame = None



def concealer_worker(image, r, g, b, intensity, out_queue) -> None:
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

        # crop = commons.moist(crop, cc-top_y,rr-top_x, 220)
        # crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, 167, 74, 34, 0.2)
        # image2 = commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 15, 5)
        crops.append(commons.apply_blur2(crop, cc-top_y, rr-top_x, b, g, r, intensity))
        bounds.append([top_x, top_y, bottom_x, bottom_y])

    Globals.concealer.crops = crops
    Globals.concealer.bounds = bounds
        # image [top_y:bottom_y, top_x:bottom_x, ] = image2

    out_queue.append({
        'index': 4,
        'crops': crops,
        'bounds': bounds
    })


def blush_worker(image, r, g, b, intensity, out_queue) -> None:
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

        crops.append(commons.apply_blur2(crop, cc-top_y, rr-top_x, b, g, r, intensity))
        bounds.append([top_x, top_y, bottom_x, bottom_y])

    Globals.blush.crops = crops
    Globals.blush.bounds = bounds

    out_queue.append({
        'index': 3,
        'crops': crops,
        'bounds': bounds
    })


def lipstick_worker(image, r, g, b, intensity, out_queue) -> None:
    crops = []
    bounds = []

    for region in constants.LIPS:
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

        crop = commons.moist(crop, cc-top_y,rr-top_x, 220)
        crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, b, g, r, intensity)
        crops.append(commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 15, 5))
        bounds.append([top_x, top_y, bottom_x, bottom_y])

    Globals.lipstick.crops = crops
    Globals.lipstick.bounds = bounds

    out_queue.append({
        'index': 0,
        'crops': crops,
        'bounds': bounds
    })


def eyeshadow_worker(image, r, g, b, intensity, out_queue) -> None:
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
        crops.append(commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 15, 5))
        bounds.append([top_x, top_y, bottom_x, bottom_y])

    Globals.eyeshadow.crops = crops
    Globals.eyeshadow.bounds = bounds

    out_queue.append({
        'index': 2,
        'crops': crops,
        'bounds': bounds
    })


def eyeliner_worker(image, r, g, b, intensity, out_queue) -> None:
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
        crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, b, g, r, intensity)
        crops.append(commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 15, 5))
        bounds.append([top_x, top_y, bottom_x, bottom_y])

    Globals.eyeliner.crops = crops
    Globals.eyeliner.bounds = bounds

    out_queue.append({
        'index': 1,
        'crops': crops,
        'bounds': bounds
    })


def foundation_worker(image, r, g, b, intensity, out_queue) -> None:
    crops = []
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

        crop = commons.moist(crop, cc-top_y,rr-top_x, 220)
        crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, b, g, r, intensity)

        crops.append(commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 15, 5))
        bounds.append([top_x, top_y, bottom_x, bottom_y])

    Globals.foundation.crops = crops
    Globals.foundation.bounds = bounds

    out_queue.append({
        'index': 5,
        'crops': crops,
        'bounds': bounds
    })


Globals.makeup_workers = {
    # 'lens_worker':          { 'function': lens_worker,          'instance': Globals.lens,       'args': [], 'enabled': False },
    'lipstick_worker':      { 'function': lipstick_worker,      'instance': Globals.lipstick,   'args': [], 'enabled': False },
    'eyeliner_worker':      { 'function': eyeliner_worker,      'instance': Globals.eyeliner,   'args': [], 'enabled': False },
    'eyeshadow_worker':     { 'function': eyeshadow_worker,     'instance': Globals.eyeshadow,  'args': [], 'enabled': False },
    'blush_worker':         { 'function': blush_worker,         'instance': Globals.blush,      'args': [], 'enabled': False },
    'concealer_worker':     { 'function': concealer_worker,     'instance': Globals.concealer,  'args': [], 'enabled': False },
    'foundation_worker':    { 'function': foundation_worker,    'instance': Globals.foundation, 'args': [], 'enabled': False },
}

def join_makeup_workers(image):
    threads = []
    shared_list = []

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
            
            for crop, [top_x, top_y, bottom_x, bottom_y] in zip(temp_img['crops'], temp_img['bounds']):
                image[top_y:bottom_y, top_x:bottom_x, ] = crop

    return image


def join_makeup_workers_static(image):
    shared_list = []

    for makeup_worker in Globals.makeup_workers:
        worker = Globals.makeup_workers[makeup_worker]

        if worker['enabled']:
            shared_list.append({
                'crops': worker['instance'].crops,
                'bounds': worker['instance'].bounds
            })

    while len(shared_list) > 0:
        temp_img = shared_list.pop()

        for crop, [top_x, top_y, bottom_x, bottom_y] in zip(temp_img['crops'], temp_img['bounds']):
                image[top_y:bottom_y, top_x:bottom_x, ] = crop

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
                print(temp)
                if temp < 500:  
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

                    face_crop = image[f_ymin:f_ymin+f_height, f_xmin:f_xmin+f_width]

                    # image.flags.writeable = False
                    results = Globals.face_mesher.process(image)
                    # image.flags.writeable = True
        
                    if results.multi_face_landmarks:
                        for landmark_list in results.multi_face_landmarks:
                
                            image_rows, image_cols, _ = image.shape
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
            
            ## CROP HERE

                        image = join_makeup_workers(image)

                        Globals.motion_detected = False

            else:
                print('no motion')
                face_crop = image[Globals.f_ymin:Globals.f_ymin+Globals.f_height, Globals.f_xmin:Globals.f_xmin+Globals.f_width]
                image = join_makeup_workers_static(image)
            # uncrop here


            Globals.prev_frame = gray.copy()

            # return image

            (flag, encodedImage) = cv2.imencode(".png", image)
            
            # ensure the frame was successfully encoded
            if not flag:
                continue
            # yield the output frame in the byte format
            yield (b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')

        # for region in constants.LIPS:
        #     roi_x = []
        #     roi_y = []
        #     for point in region:
        #         roi_x.append(Globals.idx_to_coordinates[point][0])
        #         roi_y.append(Globals.idx_to_coordinates[point][1])

        #     margin = 40
        #     top_x = min(roi_x)-margin
        #     top_y = min(roi_y)-margin
        #     bottom_x = max(roi_x)+margin
        #     bottom_y = max(roi_y)+margin

        #     rr, cc = polygon(roi_x, roi_y)
            
        #     crop = image [top_y:bottom_y, top_x:bottom_x, ]

        #     crop = commons.moist(crop, cc-top_y,rr-top_x, 220)
        #     crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, 167, 74, 34, 0.2)
        #     image2 = commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 15, 5)
        #     # image2 = commons.apply_blur2(crop,cc-top_y,rr-top_x,87, 51, 36, 1)

        #     image [top_y:bottom_y, top_x:bottom_x, ] = image2

            


        # return image

# Globals.cap.release()


def enable_makeup(makeup_type='', r=0, g=0, b=0, intensity=.7, lipstick_type='hard', gloss=False, k_h=81, k_w=81):
    Globals.motion_detected = True
    
    if makeup_type == 'eyeshadow':
        Globals.makeup_workers['eyeshadow_worker']['args'] = [r, g, b, intensity]
        Globals.makeup_workers['eyeshadow_worker']['enabled'] = True
    elif makeup_type == 'lipstick':
        Globals.makeup_workers['lipstick_worker']['args'] = [r, g, b, intensity]
        Globals.makeup_workers['lipstick_worker']['enabled'] = True
    elif makeup_type == 'blush':
        Globals.makeup_workers['blush_worker']['args'] = [r, g, b, intensity]
        Globals.makeup_workers['blush_worker']['enabled'] = True
    elif makeup_type == 'concealer':
        Globals.makeup_workers['concealer_worker']['args'] = [r, g, b, intensity]
        Globals.makeup_workers['concealer_worker']['enabled'] = True
    elif makeup_type == 'foundation':
        Globals.makeup_workers['foundation_worker']['args'] = [r, g, b, intensity]
        Globals.makeup_workers['foundation_worker']['enabled'] = True
    elif makeup_type == 'eyeliner':
        Globals.makeup_workers['eyeliner_worker']['args'] = [r, g, b, intensity]
        Globals.makeup_workers['eyeliner_worker']['enabled'] = True
    elif makeup_type == 'lens':
        Globals.makeup_workers['lens_worker']['args'] = [r, g, b]
        Globals.makeup_workers['lens_worker']['enabled'] = True


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
        Globals.makeup_workers['foundation_worker']['enabled_first'] = False
    elif makeup_type == 'eyeliner':
        Globals.makeup_workers['eyeliner_worker']['enabled'] = False
    elif makeup_type == 'lens':
        Globals.makeup_workers['lens_worker']['enabled'] = False


def start_cam():
    Globals.cap.open(0)

def stop_cam():
    Globals.cap.release()