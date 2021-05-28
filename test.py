import cv2
import src.cv.makeup.utils as mutils

# mutils.start_cam()
mutils.start_cam()
mutils.enable_makeup('eyeshadow', 34, 74, 162, .1)
mutils.enable_makeup('blush', 87, 36, 51, .4)
# mutils.enable_makeup('eyeliner', 142, 30, 29, .5)
mutils.enable_makeup('lipstick', 34, 74, 167, .6)
# mutils.enable_makeup('concealer', 87, 51, 36, 1)
# mutils.enable_makeup('foundation', 255, 253, 208, .3)
# mutils.enable_makeup('lens', 74, 136, 237)
while mutils.Globals.cap.isOpened():
    cv2.imshow("Frame", mutils.apply_makeup())
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break