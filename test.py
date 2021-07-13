################################ Read from MS HD 3000


# import cv2

# cap = cv2.VideoCapture(3)

# if not cap.isOpened():
#     print('ERROR! Unable to open camera')
# else:
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# while True:
#     _, frame = cap.read()
#     cv2.imshow("MS HD 3000", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

################################ Read from video

# import cv2
# import numpy as np

# cap = cv2.VideoCapture('video.mp4')

# Check if camera opened successfully
# if (cap.isOpened()): 
#     ret, frame = cap.read()
#     if ret:
#         print(frame.shape)
#         cv2.imshow('frame', frame)
#         cv2.waitKey(0)

# # Read until video is completed
# while(cap.isOpened()):
#   # Capture frame-by-frame
#   ret, frame = cap.read()
#   if ret == True:

#     # Display the resulting frame
#     cv2.imshow('Frame',frame)

#     # Press Q on keyboard to  exit
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#       break

#   # Break the loop
#   else: 
#     break

# # When everything done, release the video capture object
# cap.release()

# # Closes all the frames
# cv2.destroyAllWindows()

################################################################# Read from webcam with flip

# """
# Simply display the contents of the webcam with optional mirroring using OpenCV 
# via the new Pythonic cv2 interface.  Press <esc> to quit.
# """

# import cv2


# def show_webcam(mirror=False):
#     cam = cv2.VideoCapture(0)
#     while True:
#         ret_val, img = cam.read()
#         if mirror: 
#             img = cv2.flip(img, 1)
#         cv2.imshow('my webcam', img)
#         if cv2.waitKey(1) == 27: 
#             break  # esc to quit
#     cv2.destroyAllWindows()


# def main():
#     show_webcam(mirror=True)


# if __name__ == '__main__':
#     main()


################################################################

import src.cv.makeup.utils as mutils

mutils.enable_makeup('lipstick', 142, 30, 29)
mutils.enable_makeup('eyeshadow', 25, 30, 140, .3)

mutils.apply_makeup_video('video.mp4')

