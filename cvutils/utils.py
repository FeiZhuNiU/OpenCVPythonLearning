import cv2
from enum import Enum, unique
import numpy as np


# unique帮助检查没有重复的值
@unique
class BinaryMethod(Enum):
    GLOBAL = 0
    GLOBAL_INV = 1
    ADAPTIVE_MEAN = 2
    ADAPTIVE_GAUSSIAN = 3
    OTSU = 4


@unique
class SmoothMethod(Enum):
    BLUR = 0
    GAUSSIAN = 1
    MEDIAN = 2
    BILATERAL = 3


@unique
class EdgeMethod(Enum):
    SOBEL = 0
    SHERR = 1
    LAPLACIAN = 2
    CANNY = 3


def convert2binary(img, method=BinaryMethod.ADAPTIVE_GAUSSIAN, threshold=127, max_val_to_set=255, ksize=3):
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    if method == BinaryMethod.GLOBAL:
        _, ret = cv2.threshold(gray_img, thresh=threshold, maxval=max_val_to_set, type=cv2.THRESH_BINARY)
        return ret
    elif method == BinaryMethod.GLOBAL_INV:
        _, ret = cv2.threshold(gray_img, thresh=threshold, maxval=max_val_to_set, type=cv2.THRESH_BINARY_INV)
        return ret
    elif method == BinaryMethod.ADAPTIVE_MEAN:
        return cv2.adaptiveThreshold(gray_img,
                                     maxValue=max_val_to_set,
                                     adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                     thresholdType=cv2.THRESH_BINARY,
                                     blockSize=ksize,
                                     C=0)
    elif method == BinaryMethod.ADAPTIVE_GAUSSIAN:
        return cv2.adaptiveThreshold(gray_img,
                                     maxValue=max_val_to_set,
                                     adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     thresholdType=cv2.THRESH_BINARY,
                                     blockSize=ksize,
                                     C=0)
    elif method == BinaryMethod.OTSU:
        return cv2.threshold(gray_img, 0, max_val_to_set, cv2.THRESH_OTSU)


def smoothing(img, method=SmoothMethod.GAUSSIAN, ksize=3, sigma=0, border_type=None, d=0, color_sigma=80,
              space_sigma=80):
    if method == SmoothMethod.BLUR:
        return cv2.blur(img, (ksize, ksize))
    elif method == SmoothMethod.GAUSSIAN:
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)
    elif method == SmoothMethod.MEDIAN:
        return cv2.medianBlur(img, ksize)
    elif method == SmoothMethod.BILATERAL:
        return cv2.bilateralFilter(img, d, color_sigma, space_sigma, borderType=border_type)


def edge_detection(img, method=EdgeMethod.CANNY, ksize=3, low_thresh=150, high_thresh=250):
    if method == EdgeMethod.SOBEL:
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        return cv2.convertScaleAbs(sobel)
    elif method == EdgeMethod.SHERR:
        sherr = cv2.addWeighted(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1), 0.5,
                                cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1), 0.5, 0)
        return cv2.convertScaleAbs(sherr)
    elif method == EdgeMethod.LAPLACIAN:
        laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
        return cv2.convertScaleAbs(laplacian)
    elif method == EdgeMethod.CANNY:
        return cv2.Canny(img, low_thresh, high_thresh)


def detect_contour(gray_img, img_to_draw, max_n_contours=1, color=(0, 255, 0), thickness=1):
    processed_img, contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours is not None:
        longest_contour_len = []
        for contour in contours:
            if len(longest_contour_len) < max_n_contours:
                longest_contour_len.append(len(contour))
                longest_contour_len.sort(reverse=True)
            else:
                if len(contour) > min(longest_contour_len):
                    longest_contour_len.pop()
                    longest_contour_len.append(len(contour))
                    longest_contour_len.sort(reverse=True)

        longest_contour = list(filter(lambda x: len(x) >= min(longest_contour_len), contours))
        # longest_contour = list(map(lambda x: cv2.approxPolyDP(x, 0.02 * cv2.arcLength(x, True), True), longest_contour))
        longest_contour = list(map(lambda x: cv2.convexHull(x), longest_contour))
        cv2.drawContours(img_to_draw, longest_contour, -1, color, thickness)


def stroke_edges(src, dst, blur_ksize=7, edge_ksize=5):
    if blur_ksize >= 3:
        blurred_src = cv2.medianBlur(src, blur_ksize)
        gray_src = cv2.cvtColor(blurred_src, cv2.COLOR_BGR2GRAY)
    else:
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(gray_src, cv2.CV_8U, gray_src, ksize=edge_ksize)
    normalized_inverse_alpha = (1.0 / 255) * (255 - gray_src)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalized_inverse_alpha
    cv2.merge(channels, dst)


class VConvolutionFilter(object):
    def __index__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        cv2.filter2D(src, -1, self._kernel, dst)


class SharpenFilter(VConvolutionFilter):
    def __index__(self):
        kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])
        VConvolutionFilter.__init__(self, kernel)


class FindEdgeFilter(VConvolutionFilter):
    def __index__(self):
        kernel = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])
        VConvolutionFilter.__init__(self, kernel)


class BlurFilter(VConvolutionFilter):
    def __index__(self):
        kernel = np.ones((5, 5)) * 0.04
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    def __index__(self):
        kernel = np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])
        VConvolutionFilter.__init__(self, kernel)


test = [1, 2, 3, 5, 7, 0]
test.sort(reverse=True)
print(test)
test.pop()
print(test)
