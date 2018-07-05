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

video = cv2.VideoCapture("street.mp4")

print(video.get(cv2.CAP_PROP_FPS))
print(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(video.get(cv2.CAP_PROP_FRAME_WIDTH))

success, frame = video.read()
while success and cv2.waitKey(1) & 0xFF != ord('q'):
    cv2.imshow('frame', frame)
    success, frame = video.read()


# video = cv2.VideoCapture("street.mp4")
# while video.isOpened():
#     success, frame = video.read()
#     if success and cv2.waitKey(1) & 0xFF != ord('q'):
#         cv2.imshow('frame', frame)
#     else:
#         break
cv2.destroyAllWindows()
video.release()

cv2.waitKey()
