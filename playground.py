import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread('laugh.jpg', 0)
# img_filtered = cv2.bilateralFilter(img, 21, 75, 75)
# # laplacian = cv2.Laplacian(img, cv2.CV_64F)
# # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
# # sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)
# # sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# # sherr = cv2.addWeighted(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1), 0.5, cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1),
# #                         0.5, 0)
#
# canny = cv2.Canny(img, 100, 200)
# canny_2 = cv2.Canny(img_filtered, 50, 150)
# print(canny.dtype)
# threshold, laplacian = cv2.threshold(cv2.convertScaleAbs(laplacian), 0, 255, cv2.THRESH_OTSU)
#
#
# threshold, sobelxy = cv2.threshold(cv2.convertScaleAbs(sobelxy), 0, 255, cv2.THRESH_OTSU)
# sobelx = cv2.convertScaleAbs(sobelx)
# sobely = cv2.convertScaleAbs(sobely)
# sobel = cv2.convertScaleAbs(sobel)
# sherr = cv2.convertScaleAbs(sherr)

# titles = ['Original', 'Laplacian', 'Sobel dx=1', 'Sobel dy=1', 'Sobel', 'Sobel dx=dy=1', 'Sherr', 'canny']
# titles = ['Original', 'canny', 'canny_2']
# # imgs = [img, laplacian, sobelx, sobely, sobel, sobelxy, sherr, canny]
# imgs = [img, canny, canny_2]
#
# for i in range(3):
#     plt.subplot(2, 2, i + 1), plt.imshow(imgs[i], cmap='gray')
#     plt.title(titles[i]), plt.xticks([]), plt.yticks([])
#
# plt.show()

# # b, g, r = cv2.split(img)
# # img = cv2.merge((r, g, b))
#
# kernel_3 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
# cv2.imshow("original", img)
# dst1 = cv2.blur(img, (21, 21))
# dst2 = cv2.GaussianBlur(img, (21, 21), 0)
# dst3 = cv2.medianBlur(img, 21)
# dst4 = cv2.bilateralFilter(img, 21, 75, 75)
# dst5 = ndimage.convolve(img, kernel_3)
# titles = ["Original", "blur", "gaussian", "median", "bilateral", "ndimage"]
# imgs = [img, dst1, dst2, dst3, dst4, dst5]
# for i in range(6):
#     plt.subplot(3, 2, i + 1), plt.imshow(imgs[i], cmap='gray'), plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()
print(cv2.__version__)

img = cv2.imread('laugh.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.bilateralFilter(img, 21, 75, 75)
laplacian = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F))
_, img = cv2.threshold(laplacian, 20, 255, cv2.THRESH_BINARY)

_, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contour_1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_contour_2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
max_len = max(map(lambda x: len(x), contours))
longest_contour = list(filter(lambda x: len(x) == max_len, contours))

epsilon_1 = 0.01 * cv2.arcLength(longest_contour[0], True)
epsilon_2 = 0.02 * cv2.arcLength(longest_contour[0], True)


approx_1 = cv2.approxPolyDP(longest_contour[0], epsilon_1, True)
approx_2 = cv2.approxPolyDP(longest_contour[0], epsilon_2, True)


cv2.drawContours(img_contour_1, list([approx_1]), -1, (0, 255, 0), 2)
cv2.drawContours(img_contour_2, list([approx_2]), -1, (0, 255, 0), 2)
titles = ['Original Binary', 'approx_epsilon_smaller', 'approx_epsilon_bigger']
imgs = [img, img_contour_1, img_contour_2]
for i in range(3):
    plt.subplot(3, 1, i + 1), plt.imshow(imgs[i], cmap='gray'), plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# erosion = cv2.erode(img, kernel)
# dilation = cv2.dilate(img, kernel)
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# morph_gradient = dilation - erosion

# tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
#
# titles = ["Binary", "erosion", "dilation", "opening", "closing", "tophat", "blackhat", "gradient"]
# imgs = [img, erosion, dilation, opening, closing, tophat, blackhat, morph_gradient]
# for i in range(8):
#     plt.subplot(3, 3, i + 1), plt.imshow(imgs[i], cmap='gray'), plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()
cv2.waitKey()
