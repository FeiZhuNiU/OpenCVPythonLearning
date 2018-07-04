import cv2
import numpy as np

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

# img = cv2.imread("laugh.jpg", cv2.IMREAD_GRAYSCALE)
# cv2.imshow("laugh", img)

cv2.waitKey()
