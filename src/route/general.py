import cv2
import numpy as np

def draw_roi_border(image, top_x, top_y, bottom_x, bottom_y) -> np.ndarray:
    line_length = min(bottom_x - top_x, bottom_y - top_y) // 5
    line_width = line_length // 3
    line_color = (255, 255, 255)

    cv2.line(image, (top_x, top_y), (top_x + line_length, top_y), line_color, line_width)
    cv2.line(image, (top_x, top_y), (top_x, top_y + line_length), line_color, line_width)
    cv2.line(image, (bottom_x, top_y), (bottom_x - line_length, top_y), line_color, line_width)
    cv2.line(image, (bottom_x, top_y), (bottom_x, top_y + line_length), line_color, line_width)
    cv2.line(image, (top_x, bottom_y), (top_x + line_length, bottom_y), line_color, line_width)
    cv2.line(image, (top_x, bottom_y), (top_x, bottom_y - line_length), line_color, line_width)
    cv2.line(image, (bottom_x, bottom_y), (bottom_x - line_length, bottom_y), line_color, line_width)
    cv2.line(image, (bottom_x, bottom_y), (bottom_x, bottom_y - line_length), line_color, line_width)

    return image