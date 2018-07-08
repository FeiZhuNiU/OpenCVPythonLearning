import cv2
import numpy as np


def filter_out_white(img):
    if img is not None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 221])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        return cv2.bitwise_and(img, img, mask=mask)


def filter_out_red(img):
    if img is not None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red = np.array([156, 43, 46])
        upper_red = np.array([180, 255, 255])
        # inRange()方法返回的矩阵只包含0,1 0表示不在区间内
        mask = cv2.inRange(hsv, lower_red, upper_red)
        return cv2.bitwise_and(img, img, mask=mask)


img1 = np.array([[[2, 2, 2], [2, 2, 2]], [[2, 2, 2], [2, 2, 2]]])
img2 = np.array([[[2, 3, 3], [2, 3, 3]], [[2, 3, 5], [2, 3, 3]]])
mask = np.uint8([[1, 0], [1, 1]])
res = cv2.bitwise_or(img1, img2, mask=mask)
print(res)

cv2.waitKey()
