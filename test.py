# import cv2

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print('ERROR! Unable to open camera')
# else:
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

#     while True:
#         _, frame = cap.read()
#         cv2.imshow("MS HD 3000", frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break



import src.cv.makeup.video_utils as vutils

vutils.enable_makeup('lipstick', 142, 30, 29)
vutils.enable_makeup('eyeshadow', 25, 30, 140, .3)
vutils.apply_makeup_video('video.mp4')
