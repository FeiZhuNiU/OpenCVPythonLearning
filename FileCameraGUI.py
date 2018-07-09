import cv2
import numpy as np
import os

img_write = np.zeros((100, 100), np.uint8)
print(img_write.shape)
cv2.imwrite("black_square_gray.jpg", img_write)
img_write = cv2.cvtColor(img_write, cv2.COLOR_GRAY2BGR)
print(img_write.shape)
cv2.imwrite("black_square_color.jpg", img_write)

img_read = cv2.imread("black_square_gray.jpg")
print(img_read.shape)

img_read = cv2.imread("black_square_gray.jpg", cv2.IMREAD_GRAYSCALE)
print(img_read.shape)

img_read = cv2.imread("black_square_color.jpg")
print(img_read.shape)

img_read = cv2.imread("black_square_color.jpg", cv2.IMREAD_GRAYSCALE)
print(img_read.shape)

'''
# result
(100, 100)
(100, 100, 3)
(100, 100, 3)
(100, 100)
(100, 100, 3)
(100, 100)
'''

img = cv2.imread("laugh.jpg", cv2.IMREAD_GRAYSCALE)
print(img.shape)
byte_array = bytearray(img)
print(len(byte_array))
img2 = np.array(byte_array).reshape(720, 180)
# cv2.imshow("reshaped", img2)
"""
(360, 360)
129600
"""
print(img.shape)
print(img.size)
print(img.dtype)
"""
(360, 360)
129600
uint8
"""

# video = cv2.VideoCapture("street.mp4")
# print(video.get(cv2.CAP_PROP_FPS))
# print(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(video.get(cv2.CAP_PROP_FRAME_WIDTH))

video = cv2.VideoCapture("street.mp4")
# video = cv2.VideoCapture("outputVideo.avi")
while video.isOpened():
    if video.grab() and cv2.waitKey(1) & 0xFF != ord('q'):
        success, frame = video.retrieve()
        if success:
            cv2.imshow('frame', frame)
    else:
        break
cv2.destroyAllWindows()
video.release()

video = cv2.VideoCapture("street.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
video_writer = cv2.VideoWriter('outputVideo.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)
video = cv2.VideoCapture("street.mp4")
success, frame = video.read()
while success:
    video_writer.write(frame)
    success, frame = video.read()
video_writer.release()

cv2.waitKey()
