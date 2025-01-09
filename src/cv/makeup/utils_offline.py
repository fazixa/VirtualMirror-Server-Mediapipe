import cv2
from mediapipe.python.solutions import face_detection, face_mesh
import imutils
from src.cv.makeup import commons, constants
from skimage.draw import polygon

def apply_lipstick(user_image, r, g, b, gloss, intensity):
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

        if f_xmin < 0:
            f_xmin = 0
        elif f_ymin < 0:
            f_ymin = 0

        face_crop = user_image[f_ymin:f_ymin+f_height, f_xmin:f_xmin+f_width]
        face_height, face_width, _ = face_crop.shape
        face_crop = imutils.resize(face_crop, width=500)
        
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
            result = []
            # for intensity in intensities:
            face_crop_copy = face_crop.copy()
            for region in constants.LIPS:
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

                
                crop = face_crop_copy[top_y:bottom_y, top_x:bottom_x,]
                if gloss:
                    crop = commons.moist(crop, cc-top_y,rr-top_x, 220)
                crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, b, g, r, intensity)
                crop_makeup = commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 31, 10)
                face_crop_copy[top_y:bottom_y, top_x:bottom_x] = crop_makeup
            face_crop_copy = imutils.resize(face_crop_copy, width=face_width)
            face_c_height, face_c_width, _ = face_crop_copy.shape
            user_image[f_ymin:f_ymin+face_c_height, f_xmin:f_xmin+f_width] = face_crop_copy

        return user_image
    
def apply_eyeshadow(user_image, r, g, b, intensity):
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

        if f_xmin < 0:
            f_xmin = 0
        elif f_ymin < 0:
            f_ymin = 0


        face_crop = user_image[f_ymin:f_ymin+f_height, f_xmin:f_xmin+f_width]
        face_height, face_width, _ = face_crop.shape
        face_crop = imutils.resize(face_crop, width=500)
        
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
            result = []
            # for intensity in intensities:
            face_crop_copy = face_crop.copy()
            for region in constants.EYESHADOW:
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

                
                crop = face_crop_copy[top_y:bottom_y, top_x:bottom_x,]
                crop_colored = commons.apply_color(crop, cc-top_y,rr-top_x, b, g, r, intensity)
                crop_makeup = commons.apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 51, 20)
                face_crop_copy[top_y:bottom_y, top_x:bottom_x] = crop_makeup
            face_crop_copy = imutils.resize(face_crop_copy, width=face_width)
            face_c_height, face_c_width, _ = face_crop_copy.shape
            user_image[f_ymin:f_ymin+face_c_height, f_xmin:f_xmin+f_width] = face_crop_copy
    
    return user_image


def apply_blush(user_image, r, g, b, intensity):
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

        if f_xmin < 0:
            f_xmin = 0
        elif f_ymin < 0:
            f_ymin = 0


        face_crop = user_image[f_ymin:f_ymin+f_height, f_xmin:f_xmin+f_width]
        face_height, face_width, _ = face_crop.shape
        face_crop = imutils.resize(face_crop, width=500)
        
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
            result = []
            face_crop_copy = face_crop.copy()
            for region in constants.BLUSH:
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

                
                crop = face_crop_copy[top_y:bottom_y, top_x:bottom_x,]
                crop_makeup = commons.apply_blur_color(crop, cc-top_y,rr-top_x, b, g, r, intensity, 81, 30)
                face_crop_copy[top_y:bottom_y, top_x:bottom_x] = crop_makeup
            face_crop_copy = imutils.resize(face_crop_copy, width=face_width)
            face_c_height, face_c_width, _ = face_crop_copy.shape
            user_image[f_ymin:f_ymin+face_c_height, f_xmin:f_xmin+f_width] = face_crop_copy

    return user_image