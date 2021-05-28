from makeup.blush import Blush
from makeup.foundation import Foundation
from re import U
import cv2
import math
from pylab import mean
from skimage import color
import mediapipe as mp
import imutils
from typing import List, Tuple, Union
from skimage.draw import line, polygon, polygon_perimeter
import numpy as np
from numpy import c_
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
VISIBILITY_THRESHOLD = 0.5
PRESENCE_THRESHOLD = 0.5




cap = cv2.VideoCapture(1)


# LANDMARKS
LOWER_LIP_INNER = [308, 324, 318,402, 317, 14, 87, 178, 88, 95, 78]
LOWER_lIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 11, 267, 269, 270, 409, 291]
UPPER_LIP_INEER = [308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]

LEFT_EYEBROW_LOWER = [276, 283, 282, 295, 285]
RIGHT_EYEBROW_LOWER = [46, 53, 52, 65, 55]

LEFT_EYE_UPPER = [362, 398, 384, 385, 386, 387, 388, 466, 263]
RIGHT_EYE_UPPER = [133, 173, 157, 158, 159, 160, 161, 246, 33]

LEFT_EYELINER = [353,  260,  385,386, 387, 388, 466, 263, 249]
RIGHT_EYELINER = [124, 30, 158, 159, 160, 161, 246, 33, 7 ]

LEFT_CHEEK_BONE = [227,50, 205, 206, 92,210, 138, 215, 177 ]
RIGHT_CHEEK_BONE = [447, 280, 425, 426, 322, 430, 367, 435 , 401]

lEFT_CHEEK = [227,34, 111, 100, 36, 205, 187, 147, 137]
RIGHT_CHEEK = [447, 264, 340, 329, 266, 425, 411, 376, 366 ]

UPPER_FACE = [356, 389, 251, 332, 297,338,10,109, 67,103,54,21,162,127,    34, 143,156,70,63,105,66, 107, 9,336,296,334,293,300,383,372,264]
LOWER_FACE = [264, 447, 366, 401, 435, 367, 364, 394, 395, 369, 396, 175, 171, 140, 170, 169, 135, 138, 215, 177, 137, 227, 34,        143, 111, 118, 120,  128, 122, 6 ,   351, 357, 349,  347,340 ,372 ]
FACE = [356, 389, 251, 332, 297,338,10,109, 67,103,54,21,162,127, 234, 93, 132,58, 172, 136, 150 , 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454 ]


# POLYGONS
LOWER_LIP = LOWER_lIP_OUTER + LOWER_LIP_INNER
UPPER_LIP = UPPER_LIP_OUTER + UPPER_LIP_INEER
LEFT_EYESHADOW = LEFT_EYE_UPPER + LEFT_EYEBROW_LOWER
RIGHT_EYESHADOW = RIGHT_EYE_UPPER + RIGHT_EYEBROW_LOWER
FACE = UPPER_FACE + LOWER_FACE


# MAKEUPS
LIPS = [LOWER_LIP, UPPER_LIP]
EYESHADOW = [LEFT_EYESHADOW, RIGHT_EYESHADOW]
EYELINER = [LEFT_EYELINER, RIGHT_EYELINER]
CONCEALER = [LEFT_CHEEK_BONE, RIGHT_CHEEK_BONE]
FOUNDATION = [FACE]
BLUSH = [lEFT_CHEEK, RIGHT_CHEEK]



def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def apply_color(image, x, y, r, g, b, intensity):
    im_copy = image.copy()
    height, width = image.shape[:2]


    # converting desired parts of the original image to LAB color space
    lip_LAB = color.rgb2lab((im_copy[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)
    # calculating mean of each channel
    L, A, B = mean(lip_LAB[:, 0]), mean(lip_LAB[:, 1]), mean(lip_LAB[:, 2])
    # converting the color of the makeup to LAB
    L1, A1, B1 = color.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(
        3, )
    # applying the makeup color on image
    # L1, A1, B1 = color.rgb2lab(np.array((self.r / 255., self.g / 255., self.b / 255.)).reshape(1, 1, 3)).reshape(3, )
    G = L1 / L
    lip_LAB = lip_LAB.reshape(len(x), 1, 3)
    lip_LAB[:, :, 1:3] = intensity * np.array([A1, B1]) + (1 - intensity) * lip_LAB[:, :, 1:3]
    lip_LAB[:, :, 0] = lip_LAB[:, :, 0] * (1 + intensity * (G - 1))
    # converting back toRGB

    im_copy[x, y] = color.lab2rgb(lip_LAB).reshape(len(x), 3) * 255

    return im_copy


def moist(image, x,y, white):
    """  finds all the points fillig the lips
    Args:
        param1 : if motion is detected this parameter is the X cordinate of the lower lip otherwise the X cordinates of the previous frame gloss parts must be passed
        param2 : if motion is detected this parameter is the Y cordinate of the lower lip otherwise the Y cordinates of the previous frame gloss parts must be passed
        param3 : whether motion is detected
        param4 : red
        param5 : green
        param6 : blue

    Returns:
        42-element tuple containing

        -  (*array*): X cordinates of the pixels of the lips which must be glossy
        -  (*array*): Y cordinates of the pixels of the lips which must be glossy
            
    Todo:
        * needs major cleanup
        
    """
    intensitymoist =0.2

    val = color.rgb2lab((image[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)
    L= mean(val[:, 0])
    L1, A1, B1 = color.rgb2lab(np.array((white / 255., white / 255., white / 255.)).reshape(1, 1, 3)).reshape(3, )
    ll = L1 - L
    length = int(len(x)/4)
    Li = val[:, 0]
    light_points = sorted(Li)[-length:]
    # light_points = sorted(Li)[:length]
    min_val = min(light_points)
    max_val = max(light_points)



    for i in range(len(val[:, 0])):
        if (val[i, 0] <= max_val and val[i, 0] >=min_val):
            val[i, 0]+= ll*intensitymoist



    image2 = image.copy()
    image2[x, y] = color.lab2rgb(val).reshape(len(x), 3) * 255
    height, width = image.shape[:2]
    filter = np.zeros((height, width))
    filter[x,y] = 1
    kernel = np.ones((12, 12), np.uint8)
    filter = cv2.erode(filter, kernel, iterations=1)
    alpha = np.zeros([height, width, 3], dtype='float64')
    alpha[:, :, 0] = filter
    alpha[:, :, 1] = filter
    alpha[:, :, 2] = filter
    im_copy = (alpha * image2 + (1 - alpha) *image).astype('uint8')
    return im_copy



def apply_blur(image, image2, x, y, gussiankernel, erosionkernel):
    height, width = image.shape[:2]
    filter = np.zeros((height, width))
    filter[x,y] = 1
    # Erosion to reduce blur size
    filter = cv2.GaussianBlur(filter, (gussiankernel, gussiankernel), 0)
    kernel = np.ones((erosionkernel, erosionkernel), np.uint8)
    filter = cv2.erode(filter, kernel, iterations=1)
    alpha = np.zeros([height, width, 3], dtype='float64')
    alpha[:, :, 0] = filter
    alpha[:, :, 1] = filter
    alpha[:, :, 2] = filter
    im_copy = (alpha * image2 + (1 - alpha) *image).astype('uint8')

    return im_copy


def apply_blur2(image, x, y):
    intensity = 0.8
    r = 51
    g = 36
    b = 87
    # Create blush shape
    height, width = image.shape[:2]
    mask = np.zeros((height, width))

    mask[x,y] = 1

    mask = cv2.GaussianBlur(mask, (51, 51), 0) * intensity
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    # print(np.array(c_[x_right, y_right])[:, 0])
    val = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(float)

    val[:, :, 0] = val[:, :, 0] / 255. * 100.
    val[:, :, 1] = val[:, :, 1] - 128.
    val[:, :, 2] = val[:, :, 2] - 128.
    LAB = color.rgb2lab(np.array((float(r) / 255., float(g) / 255., float(b) / 255.)).reshape(1, 1, 3)).reshape(3, )
    mean_val = np.mean(np.mean(val, axis=0), axis=0)


    mask = np.array([mask, mask, mask])
    mask = np.transpose(mask, (1, 2, 0))

    lab = np.multiply((LAB - mean_val), mask)

    val[:, :, 0] = np.clip(val[:, :, 0] + lab[:, :, 0], 0, 100)
    val[:, :, 1] = np.clip(val[:, :, 1] + lab[:, :, 1], -127, 128)
    val[:, :, 2] = np.clip(val[:, :, 2] + lab[:, :, 2], -127, 128)

    image = (color.lab2rgb(val) * 255).astype(np.uint8)
    return image



with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    
    

    if results.multi_face_landmarks:
      for landmark_list in results.multi_face_landmarks:
 
        image_rows, image_cols, _ = image.shape
        idx_to_coordinates = {}
        for idx, landmark in enumerate(landmark_list.landmark):
            
            if ((landmark.HasField('visibility') and
                landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                landmark.presence < PRESENCE_THRESHOLD)):
                continue
            landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                        image_cols, image_rows)

            if landmark_px:
                idx_to_coordinates[idx] = landmark_px

    ## CROP HERE

    for region in CONCEALER:
        eye_x = []
        eye_y = []
        for point in region:
            eye_x.append(idx_to_coordinates[point][0])
            eye_y.append(idx_to_coordinates[point][1])

        margin = 40
        top_x = min(eye_x)-margin
        top_y = min(eye_y)-margin
        bottom_x = max(eye_x)+margin
        bottom_y = max(eye_y)+margin

        rr, cc = polygon(eye_x, eye_y)
        
        crop = image [top_y:bottom_y, top_x:bottom_x, ]

        # crop = moist(crop, cc-top_y,rr-top_x, 220)
        # crop_colored = apply_color(crop, cc-top_y,rr-top_x, 167, 74, 34, 0.2)
        # image2 = apply_blur(crop,crop_colored,cc-top_y,rr-top_x, 15, 5)
        image2 = apply_blur2(crop,cc-top_y,rr-top_x)

        image [top_y:bottom_y, top_x:bottom_x, ] = image2

        
    # uncrop here


    cv2.imshow('Makeup', image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()