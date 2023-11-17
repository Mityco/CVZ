import numpy as np
from PIL import Image
import cmath
import math
from matplotlib import pyplot as plt


Wr = np.random.normal(0, 1, [256, 256])
# std = Wr.std()
# mean = Wr.mean()
# Wr = (Wr - mean) / std
# new_std = Wr.std()
# new_mean = Wr.mean()

image = Image.open("../Lab2/bridge.tif")
arr = np.asarray(image)


def additive_dft_embed(C, CVZ, alpha):
    N1, N2 = C.shape
    L1, L2 = CVZ.shape
    CW = np.fft.fft2(C)
    polar = np.vectorize(cmath.polar)
    rect = np.vectorize(cmath.rect)
    module, phase = polar(CW.flatten())

    module = module.reshape(C.shape)
    x_coord = N1 // 2 - L1 // 2
    y_coord = N2 // 2 - L2 // 2
    module[x_coord: x_coord + L1, y_coord: y_coord + L2] = module[x_coord: x_coord + L1, y_coord: y_coord + L2] + alpha * CVZ
    CW = rect(module.flatten(), phase)
    CW = CW.reshape(C.shape)
    CW = np.fft.ifft2(CW)
    return CW


def additive_dft_extract(CW, C, L1, L2, alpha):
    N1, N2 = CW.shape
    CW = np.fft.fft2(CW)
    polar = np.vectorize(cmath.polar)
    rect = np.vectorize(cmath.rect)
    module, phase = polar(CW.flatten())
    c_module, c_phase = polar(C.flatten())
    x_coord = N1 // 2 - L1 // 2
    y_coord = N2 // 2 - L2 // 2
    module = module.reshape(CW.shape)
    c_module = c_module.reshape(CW.shape)
    CVZ = (module[x_coord: x_coord + L1, y_coord: y_coord + L2] - c_module[x_coord: x_coord + L1, y_coord: y_coord + L2]) / alpha
    return CVZ


max_ro = 0
max_alpha = 0
for i in range(3400, 3500, 1):
    CW = additive_dft_embed(arr, Wr, i)
    sigma = additive_dft_extract(CW, arr, 256, 256, i)
    flatten_wr = Wr.flatten()
    flatten_sigma = sigma.flatten()
    ro = sum(flatten_wr * flatten_sigma) / (((sum(flatten_wr ** 2)) ** (1/2)) * ((sum(flatten_sigma ** 2)) ** (1/2)))
    if ro > max_ro:
        max_ro = ro
        max_alpha = i
print(max_ro)
print(max_alpha)
