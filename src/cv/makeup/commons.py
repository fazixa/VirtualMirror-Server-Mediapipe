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
    intensitymoist =0.5

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


def apply_blur_color(image, x, y, r, g, b, intensity, gussiankernel=51, erosionkernel=15):


    """
    applies blur and color on the desired region

    Args:
        arg1 (image) : input image
        arg2 (int array) : X cordinates of the desired region
        arg3 (int array) : Y cordinates of the desired region
        arg4 : red value of rgb color
        arg5 : green value of rgb color
        arg6 : blue value of rgb color

    Returns:
        the image with the applied clor and blur on the desired region

    Note:
        this function is used when we need the color to be more faded on the edges of the region, this will give us a more natural look which is used on blush, concelaer and foundation.
    """

    # intensity = 0.8
    # r = 51
    # g = 36
    # b = 87
    # Create blush shape
    height, width = image.shape[:2]
    mask = np.zeros((height, width))

    mask[x,y] = 1

    mask = cv2.GaussianBlur(mask, (gussiankernel, gussiankernel), 0) * intensity
    kernel = np.ones((erosionkernel, erosionkernel), np.uint8)
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

