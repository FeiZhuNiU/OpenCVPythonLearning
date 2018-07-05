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
cv2.imshow("reshaped", img2)
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
cv2.waitKey()
