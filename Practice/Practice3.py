import numpy as np
import cmath
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import fft2, ifft2, dct, idct


def dct2(a):
    return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a):
    return idct(idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def psnr(C, Cw):
    return 10 * np.log10(np.power(255, 2) / np.mean(np.power((C - Cw), 2)))


img = plt.imread("bridge.tif").astype(float)
img2 = img.copy()
img2[0, 0] += 100
print(psnr(img, img2))
print(cv2.PSNR(img, img2))
result_dct2 = dct2(img)
result_idct2 = idct2(result_dct2)
print(psnr(img, result_idct2))


def simple_dct_embed(C, logo):
    N1, N2 = C.shape
    L1, L2 = logo.shape
    CW = dct2(C)
    x_coord = N1 - L1
    y_coord = N2 - L2
    CW[x_coord: x_coord + L1 + 1, y_coord: y_coord + L2 + 1] = logo * 100
    CW = idct2(CW)
    return CW


logo = plt.imread("logo.bmp").astype(float)
logo = logo[:, :, 0] // 255
CW = simple_dct_embed(img, logo)
print(psnr(img, CW))
plt.imshow(CW, cmap="gray")
plt.show()
plt.imshow(img, cmap="gray")


def simple_dct_extract(CW, L1, L2):
    N1, N2 = CW.shape
    CW = dct2(CW)
    x_coord = N1 - L1
    y_coord = N2 - L2
    logo = CW[x_coord: x_coord + L1 + 1, y_coord: y_coord + L2 + 1]

    return logo


plt.imsave('CW.png', CW)
CW = plt.imread('CW.png')[:, :, 0]
logo_extracted = simple_dct_extract(CW, logo.shape[0], logo.shape[1])
plt.imshow(logo_extracted)


def simple_dft_embed(C, logo):
    N1, N2 = C.shape
    L1, L2 = logo.shape
    CW = fft2(C)
    polar = np.vectorize(cmath.polar)
    rect = np.vectorize(cmath.rect)
    module, phase = polar(CW.flatten())

    module = module.reshape(C.shape)
    x_coord = N1 // 2 - L1 // 2
    y_coord = N2 // 2 - L2 // 2
    module[x_coord: x_coord + L1, y_coord: y_coord + L2] = logo
    CW = rect(module.flatten(), phase)
    CW = CW.reshape(C.shape)
    CW = ifft2(CW)
    return CW


def simple_dft_extract(CW, L1, L2):
    N1, N2 = CW.shape
    CW = fft2(CW)
    polar = np.vectorize(cmath.polar)
    rect = np.vectorize(cmath.rect)
    module, phase = polar(CW.flatten())
    x_coord = N1 // 2 - L1 // 2
    y_coord = N2 // 2 - L2 // 2
    module = module.reshape(CW.shape)
    logo = module[x_coord: x_coord + L1, y_coord: y_coord + L2]
    return logo


CW = simple_dft_embed(img, logo)
print(psnr(img, CW))

plt.imshow(CW.real, cmap="gray")
plt.imshow(simple_dft_extract(CW, logo.shape[0], logo.shape[1]), cmap="gray")
plt.show()
