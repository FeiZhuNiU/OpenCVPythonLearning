import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

# # shape: 360*360
# img = cv2.imread("laugh.jpg", cv2.IMREAD_GRAYSCALE)
# # 返回的是复数 dtype.complex128
# fft = np.fft.fft2(img)
# # 平移
# fftshift = np.fft.fftshift(fft)
# # 频谱 dtype.float64 magnitude_spectrum[180,180] = 329.2611
# magnitude_spectrum = 20 * np.log(np.abs(fftshift))
# # 如果想用cv2.imshow()显示
# # magnitude_spectrum_uint8 = np.uint8(255 * (magnitude_spectrum / np.max(magnitude_spectrum)))
# # cv2.imshow("magnitude_spectrum_uint8", magnitude_spectrum_uint8)
# rows, cols = img.shape
# crow, ccol = rows / 2, cols / 2
# # 频谱中心区域添加60×60的蒙板，相当于过滤了低频部分
# fftshift[int(crow - 30):int(crow + 30), int(ccol - 30):int(ccol + 30)] = 0
# magnitude_spectrum_filter = 20 * np.log(np.abs(fftshift))
# # 中心平移回到左上角
# f_ishift = np.fft.ifftshift(fftshift)
# # 使用FFT逆变换，结果是复数
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.abs(img_back)
# # img_back_uint8 = np.uint8(255 * (img_back / np.max(img_back)))
# # cv2.imshow("img_back_uint8", img_back_uint8)
# # plt.subplot(221)
# # plt.imshow(img, cmap='gray')
# # plt.title('laugh.jpg')
# # # 省略x,y坐标
# # plt.xticks([]), plt.yticks([])
# # plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
# # plt.title('magnitude_spectrum'), plt.xticks([]), plt.yticks([])
# # plt.subplot(223), plt.imshow(magnitude_spectrum_filter, cmap='gray')
# # plt.title('High Pass Filter'), plt.xticks([]), plt.yticks([])
# # plt.subplot(224), plt.imshow(img_back, cmap='gray')
# # plt.title('High Pass Result'), plt.xticks([]), plt.yticks([])
# # plt.show()
#
#
# # img = cv2.imread('laugh.jpg', cv2.IMREAD_GRAYSCALE)
# # dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
# # # 平移还是要靠numpy
# # dft_shift = np.fft.fftshift(dft)
# # magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
# # print(dft.dtype)
# # print(dft_shift.dtype)
# #
# # rows, cols = img.shape
# # crow, ccol = int(rows / 2), int(cols / 2)
# #
# # # create a mask first, center square is 1, remaining all zeros
# # mask = np.ones((rows, cols, 2), np.uint8)
# # mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
# #
# # # apply mask and inverse DFT
# # fshift = dft_shift * mask
# # f_ishift = np.fft.ifftshift(fshift)
# # img_back = cv2.idft(f_ishift)
# # img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
# #
# # plt.subplot(311), plt.imshow(img, cmap='gray')
# # plt.title('laugh.jpg'), plt.xticks([]), plt.yticks([])
# # plt.subplot(312), plt.imshow(magnitude_spectrum, cmap='gray')
# # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# # plt.subplot(313), plt.imshow(img_back, cmap='gray')
# # plt.title('result'), plt.xticks([]), plt.yticks([])
# #
# # plt.show()
#
# img = cv2.imread('laugh.jpg', cv2.IMREAD_GRAYSCALE)
# rows, cols = img.shape
# nrows = cv2.getOptimalDFTSize(rows)
# ncols = cv2.getOptimalDFTSize(cols)
# nimg = np.zeros((nrows, ncols))
# nimg[:rows, :cols] = img

img = cv2.imread("laugh.jpg", cv2.IMREAD_GRAYSCALE)
kernel_3 = np.array(
    [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ]
)
kernel_5 = np.array(
    [
        [-1, -1, -1, -1, -1],
        [-1, 1, 2, 1, -1],
        [-1, 2, 4, 2, -1],
        [-1, 1, 2, 1, -1],
        [-1, -1, -1, -1, -1]
    ]
)
dst_3 = cv2.filter2D(img, -1, kernel=kernel_3)
dst_5 = cv2.filter2D(img, -1, kernel=kernel_5)
plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(dst_3, cmap='gray'), plt.title('Kernel_3 Result')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(dst_5, cmap='gray'), plt.title('Kernel_5 Result')
plt.xticks([]), plt.yticks([])
plt.show()


cv2.waitKey()
dst_3_scipy = ndimage.convolve(img, kernel_5)
plt.subplot(223), plt.imshow(dst_3_scipy, cmap='gray'), plt.title('Kernal_3 scipy Result')
plt.xticks([]), plt.yticks([])
