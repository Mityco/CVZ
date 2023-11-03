import numpy as np
from PIL import Image
from cmath import phase
import math
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import os


def threshold_processing(x):
    if x > 0.8:
        x = 1
    else:
        x = 0
    return x


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def auto_selection(image):
    psnr = np.inf
    best_alpha = 0
    best_p = 0
    for alpha in range(-1001, 1000000, 100):
        image_array = np.asarray(image)
        spectre_array = np.fft.fft2(image_array)

        get_phase = np.vectorize(phase)
        phase_array = get_phase(spectre_array)
        abs_spectre = abs(spectre_array)
        abs_spectre1 = abs(spectre_array)
        changed_abs_spectre = abs_spectre
        changed_abs_spectre[128:384, 128:384] = abs_spectre[128:384, 128:384] + alpha * CVZ
        changed_spectre = changed_abs_spectre * np.exp(phase_array * 1j)

        reverse_array = abs(np.fft.ifft2(changed_spectre))

        reverse_spectre_array = np.fft.fft2(reverse_array)
        reverse_abs_spectre = abs(reverse_spectre_array / np.exp(phase_array * 1j))
        included_cvz = (reverse_abs_spectre[128:384, 128:384] - abs_spectre1[128:384, 128:384]) / alpha
        flatten_cvz = CVZ.flatten()
        flatten_included_cvz = included_cvz.flatten()
        p = sum(flatten_cvz * flatten_included_cvz) / (
                    ((sum(flatten_cvz ** 2)) ** (1 / 2)) * ((sum(flatten_included_cvz ** 2)) ** (1 / 2)))

        included_cvz_estimation = threshold_processing(p)
        if included_cvz_estimation:
            new_psnr = PSNR(image_array, reverse_array)
            if new_psnr < psnr:
                psnr = new_psnr
                best_alpha = alpha
                best_p = p
        if alpha%10000 == 999:
            print(best_p)

    return best_alpha, psnr, best_p


def generate_false_detection_cvz(count):
    false_detection_cvz = []
    for i in range(count):
        false_detection_cvz.append(np.random.random(65536))
    return false_detection_cvz


def proximity_function(first_cvz, second_cvz):
    return sum(first_cvz * second_cvz) / (
            ((sum(first_cvz ** 2)) ** (1 / 2)) * ((sum(second_cvz ** 2)) ** (1 / 2)))


def false_detection(false_detection_cvz, cvz):
    false_detection_proximity_array = []
    for false_cvz in false_detection_cvz:
        false_detection_proximity_array.append(proximity_function(cvz, false_cvz))
    return false_detection_proximity_array


alpha1 = 10000
CVZ = np.random.normal(0.5, 0.5, size=[256, 256])
flatten_CVZ = CVZ.flatten()
false_detection_cvz = generate_false_detection_cvz(100)
false_detection_proximity_array = false_detection(false_detection_cvz, CVZ.flatten())

x = np.arange(0, 100, 1)
y = false_detection_proximity_array
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(x, y, color="red")
plt.show()

image = Image.open("bridge.tif")

# print(auto_selection(image))

# image.show()
image_array = np.asarray(image)

spectre_array = np.fft.fft2(image_array)

get_phase = np.vectorize(phase)
phase_array = get_phase(spectre_array)
abs_spectre = abs(spectre_array)
abs_spectre1 = abs(spectre_array)
changed_abs_spectre = abs_spectre
changed_abs_spectre[128:384, 128:384] = abs_spectre[128:384, 128:384] + alpha1*CVZ
changed_spectre = changed_abs_spectre * np.exp(phase_array*1j)

# test_abs = abs(changed_spectre / np.exp(phase_array*1j))
# test_cvz = (test_abs[128:384, 128:384] - abs_spectre1[128:384, 128:384]) / alpha
# spectre_image = Image.fromarray(spectre_array)
reverse_array = abs(np.fft.ifft2(changed_spectre))
save_reverse_array = reverse_array
reverse_spectre_array = np.fft.fft2(reverse_array)
reverse_abs_spectre = abs(reverse_spectre_array / np.exp(phase_array*1j))
included_cvz = (reverse_abs_spectre[128:384, 128:384] - abs_spectre1[128:384, 128:384]) / alpha1
flatten_cvz = CVZ.flatten()
flatten_included_cvz = included_cvz.flatten()
p = sum(flatten_cvz*flatten_included_cvz) / (((sum(flatten_cvz**2))**(1/2)) * ((sum(flatten_included_cvz**2))**(1/2)))

included_cvz_estimation = threshold_processing(p)
print(p)
print(included_cvz_estimation)

reverse_image = Image.fromarray(reverse_array)
reverse_image.show()


# CUT

replacement_proportion = 0.25
reverse_array[0:len(reverse_array), 0:int(replacement_proportion*len(reverse_array))] = image_array[0:len(image_array):, 0:int(replacement_proportion*len(image_array))]
reverse_spectre_array = np.fft.fft2(reverse_array)
reverse_abs_spectre = abs(reverse_spectre_array / np.exp(phase_array*1j))
cut_cvz = (reverse_abs_spectre[128:384, 128:384] - abs_spectre1[128:384, 128:384]) / alpha1
flatten_cvz = CVZ.flatten()
flatten_cut_cvz = cut_cvz.flatten()
p = sum(flatten_cvz*flatten_cut_cvz) / (((sum(flatten_cvz**2))**(1/2)) * ((sum(flatten_cut_cvz**2))**(1/2)))

included_cvz_estimation = threshold_processing(p)
print(p)
print(included_cvz_estimation)

cut_image = Image.fromarray(reverse_array)
cut_image.show()


# ROTATION

rotation_angle = 90

rotated_image = reverse_image.rotate(rotation_angle)
rotated_image.show()
rotated_image_array = np.asarray(rotated_image)
spectre_array = np.fft.fft2(rotated_image_array)

reverse_array = abs(np.fft.ifft2(spectre_array))
reverse_spectre_array = np.fft.fft2(reverse_array)
reverse_abs_spectre = abs(reverse_spectre_array / np.exp(phase_array*1j))
rotated_cvz = (reverse_abs_spectre[128:384, 128:384] - abs_spectre1[128:384, 128:384]) / alpha1
flatten_cvz = CVZ.flatten()
flatten_rotated_cvz = rotated_cvz.flatten()
p = sum(flatten_cvz*flatten_rotated_cvz) / (((sum(flatten_cvz**2))**(1/2)) * ((sum(flatten_rotated_cvz**2))**(1/2)))

included_cvz_estimation = threshold_processing(p)
print(p)
print(included_cvz_estimation)


# SMOOTH

window = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9

smooth_array = convolve2d(reverse_image, window, boundary="symm", mode="same")
smooth_image = Image.fromarray(smooth_array)
smooth_image.show()

spectre_array = np.fft.fft2(smooth_array)

reverse_array = abs(np.fft.ifft2(spectre_array))
reverse_spectre_array = np.fft.fft2(reverse_array)
reverse_abs_spectre = abs(reverse_spectre_array / np.exp(phase_array*1j))
rotated_cvz = (reverse_abs_spectre[128:384, 128:384] - abs_spectre1[128:384, 128:384]) / alpha1
flatten_cvz = CVZ.flatten()
flatten_smoothed_cvz = rotated_cvz.flatten()
p = sum(flatten_cvz*flatten_smoothed_cvz) / (((sum(flatten_cvz**2))**(1/2)) * ((sum(flatten_smoothed_cvz**2))**(1/2)))

included_cvz_estimation = threshold_processing(p)
print(p)
print(included_cvz_estimation)


# JPEG

rgb_reverse_image = reverse_image.convert("RGB")
rgb_reverse_image.save("JPEG_image.jpg")

JPEG_image = Image.open("JPEG_image.jpg").convert("L")
JPEG_image.show()

JPEG_array = np.asarray(JPEG_image)

spectre_array = np.fft.fft2(JPEG_array)

reverse_array = abs(np.fft.ifft2(spectre_array))
reverse_spectre_array = np.fft.fft2(reverse_array)
reverse_abs_spectre = abs(reverse_spectre_array / np.exp(phase_array*1j))
rotated_cvz = (reverse_abs_spectre[128:384, 128:384] - abs_spectre1[128:384, 128:384]) / alpha1
flatten_cvz = CVZ.flatten()
flatten_JPEG_cvz = rotated_cvz.flatten()
p = sum(flatten_cvz*flatten_JPEG_cvz) / (((sum(flatten_cvz**2))**(1/2)) * ((sum(flatten_JPEG_cvz**2))**(1/2)))

included_cvz_estimation = threshold_processing(p)
print(p)
print(included_cvz_estimation)


print()
