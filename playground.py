import cv2
from cvutils import utils
import numpy as np
from matplotlib import pyplot as plt
print(cv2.__version__)
img = cv2.imread('forGrab.jpg')
img_bin = utils.convert2binary(utils.smoothing(img, method=utils.SmoothMethod.GAUSSIAN, ksize=3),
                               method=utils.BinaryMethod.GLOBAL_INV, ksize=3)
out_rect = utils.find_rectangle_outer_contour(img_bin)
print(out_rect)
cv2.rectangle(img, (out_rect[0], out_rect[1]), (out_rect[0] + out_rect[2], out_rect[1] + out_rect[3]), (0, 255, 0))
mask = np.ones(img.shape[:2], np.uint8)
cv2.grabCut(img, mask, out_rect, None, None, 3, cv2.GC_INIT_WITH_RECT)
# cv2.drawContours(img, list([outer_contour]), 0, (0, 255, 0))
cv2.imshow("origin", img)
aa = np.array([[0, 1], [1, 0]])
aa = aa[:, :, np.newaxis]
print(aa)
print(mask.shape)
cv2.imshow("mask", cv2.equalizeHist(mask))
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
print(mask2.shape)
img = img * mask2[:, :, np.newaxis]
cv2.imshow("result", img)
# print(img.shape)
# mask = np.zeros(img.shape[:2], np.uint8)
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)
# rect = (10, 150, img.shape[0]-10, img.shape[1]-50)
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# img = img * mask2[:, :, np.newaxis]
# plt.imshow(img), plt.colorbar(), plt.show()


#
# cv2.imshow("binary", img_bin)
#
# closing = utils.closing(img_bin, iterations=5)
#
# cv2.imshow("closing", closing)
#
# sure_background = cv2.dilate(closing, np.ones((3, 3), np.uint8), iterations=3)
#
# cv2.imshow("background", sure_background)
#
# dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
# ret, sure_fg = cv2.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)
# cv2.imshow("foreground", sure_fg)
#
# cv2.grabCut()
# remove noises

# # img = np.array([
# #     [0, 255, 0, 0],
# #     [0, 0, 0, 255],
# #     [0, 0, 0, 255],
# #     [255, 0, 0, 0]
# #
# # ], np.uint8)
# img = cv2.imread('laugh.jpg', 0)
# img_filtered = cv2.bilateralFilter(img, 21, 75, 75)
# img_bin = utils.convert2binary(img_filtered, ksize=91)
# # _, labels = cv2.connectedComponents(img)
# # # print(labels)
# #
# # _, labels_2, stats, centroids = cv2.connectedComponentsWithStats(img)
# # print(labels_2)
# # print(stats)
# # print(centroids)
# #
# # res = cv2.equalizeHist(cv2.convertScaleAbs(labels))
# # # print(res)
# # cv2.imshow("bin", res)
# # # print(labels.dtype)
# dist_img = cv2.distanceTransform(img_bin, cv2.DIST_L1, cv2.DIST_MASK_3)
#
# plt.subplot(1, 2, 1), plt.imshow(dist_img)
# plt.title("jet dist"), plt.xticks([]), plt.yticks([])
# plt.subplot(1, 2, 2), plt.imshow(cv2.equalizeHist(cv2.convertScaleAbs(dist_img)), cmap='gray')
# plt.title("gray_dist"), plt.xticks([]), plt.yticks([])
# plt.show()
# # # laplacian = cv2.Laplacian(img, cv2.CV_64F)
# # # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# # # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
# # # sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)
# # # sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# # # sherr = cv2.addWeighted(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1), 0.5, cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1),
# # #                         0.5, 0)
# #
# # canny = cv2.Canny(img, 100, 200)
# # canny_2 = cv2.Canny(img_filtered, 50, 150)
# # print(canny.dtype)
# # threshold, laplacian = cv2.threshold(cv2.convertScaleAbs(laplacian), 0, 255, cv2.THRESH_OTSU)
# #
# #
# # threshold, sobelxy = cv2.threshold(cv2.convertScaleAbs(sobelxy), 0, 255, cv2.THRESH_OTSU)
# # sobelx = cv2.convertScaleAbs(sobelx)
# # sobely = cv2.convertScaleAbs(sobely)
# # sobel = cv2.convertScaleAbs(sobel)
# # sherr = cv2.convertScaleAbs(sherr)
#
# # titles = ['Original', 'Laplacian', 'Sobel dx=1', 'Sobel dy=1', 'Sobel', 'Sobel dx=dy=1', 'Sherr', 'canny']
# # titles = ['Original', 'canny', 'canny_2']
# # # imgs = [img, laplacian, sobelx, sobely, sobel, sobelxy, sherr, canny]
# # imgs = [img, canny, canny_2]
# #
# # for i in range(3):
# #     plt.subplot(2, 2, i + 1), plt.imshow(imgs[i], cmap='gray')
# #     plt.title(titles[i]), plt.xticks([]), plt.yticks([])
# #
# # plt.show()
#
# # # b, g, r = cv2.split(img)
# # # img = cv2.merge((r, g, b))
# #
# # kernel_3 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
# # cv2.imshow("original", img)
# # dst1 = cv2.blur(img, (21, 21))
# # dst2 = cv2.GaussianBlur(img, (21, 21), 0)
# # dst3 = cv2.medianBlur(img, 21)
# # dst4 = cv2.bilateralFilter(img, 21, 75, 75)
# # dst5 = ndimage.convolve(img, kernel_3)
# # titles = ["Original", "blur", "gaussian", "median", "bilateral", "ndimage"]
# # imgs = [img, dst1, dst2, dst3, dst4, dst5]
# # for i in range(6):
# #     plt.subplot(3, 2, i + 1), plt.imshow(imgs[i], cmap='gray'), plt.title(titles[i])
# #     plt.xticks([]), plt.yticks([])
# # plt.show()
# print(cv2.__version__)
#
# img = cv2.imread('laugh.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.bilateralFilter(img, 21, 75, 75)
# laplacian = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F))
# _, img = cv2.threshold(laplacian, 20, 255, cv2.THRESH_BINARY)
#
# _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# img_contour_1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# img_contour_2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# img_contour_3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# max_len = max(map(lambda x: len(x), contours))
# longest_contour = list(filter(lambda x: len(x) == max_len, contours))
#
# epsilon_1 = 0.01 * cv2.arcLength(longest_contour[0], True)
# epsilon_2 = 0.02 * cv2.arcLength(longest_contour[0], True)
#
#
# approx_1 = cv2.approxPolyDP(longest_contour[0], epsilon_1, True)
# approx_2 = cv2.approxPolyDP(longest_contour[0], epsilon_2, True)
# hull = cv2.convexHull(longest_contour[0])
#
# cv2.drawContours(img_contour_1, list([approx_1]), -1, (0, 255, 0), 2)
# cv2.drawContours(img_contour_2, list([approx_1]), -1, (0, 255, 0), 2)
# cv2.drawContours(img_contour_3, list([hull]), -1, (0, 255, 0), 2)
#
# titles = ['Original Binary', 'approx_epsilon_smaller', 'approx_epsilon_bigger', 'hull']
# imgs = [img, img_contour_1, img_contour_2, img_contour_3]
# for i in range(4):
#     plt.subplot(2, 2, i + 1), plt.imshow(imgs[i], cmap='gray'), plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()
#
# # erosion = cv2.erode(img, kernel)
# # dilation = cv2.dilate(img, kernel)
# # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# # closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# # morph_gradient = dilation - erosion
#
# # tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# # blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
# #
# # titles = ["Binary", "erosion", "dilation", "opening", "closing", "tophat", "blackhat", "gradient"]
# # imgs = [img, e/rosion, dilation, opening, closing, tophat, blackhat, morph_gradient]
# # for i in range(8):
# #     plt.subplot(3, 3, i + 1), plt.imshow(imgs[i], cmap='gray'), plt.title(titles[i])
# #     plt.xticks([]), plt.yticks([])
# # plt.show()
cv2.waitKey()
