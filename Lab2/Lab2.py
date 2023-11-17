import numpy as np
from PIL import Image
from cmath import phase
import math
from matplotlib import pyplot as plt


def threshold_processing(x):
    if x > 0.7:
        x = 1
    else:
        x = 0
    return x


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))


def auto_selection(image):
    psnr = np.inf
    best_alpha = 0
    best_p = 0
    for alpha in range(-1001, 1000000, 1000):
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
            best_p = p
            new_psnr = PSNR(image_array, reverse_array)
            # if new_psnr < psnr:
            #     psnr = new_psnr
            #     best_alpha = alpha
        if alpha % 10000 == 999:
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


alpha1 = 100
CVZ = np.random.normal(0, 1, (256, 256))
flatten_CVZ = CVZ.flatten()
false_detection_cvz = generate_false_detection_cvz(100)
false_detection_proximity_array = false_detection(false_detection_cvz, CVZ.flatten())

x = np.arange(0, 100, 1)
y = false_detection_proximity_array
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(x, y, color="red")
plt.show()

image = Image.open("../Lab3/bridge.tif")

#print(auto_selection(image))

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

reverse_image = Image.fromarray(reverse_array)
reverse_image.convert("RGB").save("img_with_cvz.png")
new_image = Image.open("img_with_cvz.png").convert("L")

reverse_array = np.asarray(new_image)
reverse_spectre_array = np.fft.fft2(reverse_array)
reverse_abs_spectre = abs(reverse_spectre_array / np.exp(phase_array*1j))
included_cvz = (reverse_abs_spectre[128:384, 128:384] - abs_spectre1[128:384, 128:384]) / alpha1
flatten_cvz = CVZ.flatten()
flatten_included_cvz = included_cvz.flatten()
p = sum(flatten_cvz*flatten_included_cvz) / (((sum(flatten_cvz**2))**(1/2)) * ((sum(flatten_included_cvz**2))**(1/2)))

included_cvz_estimation = threshold_processing(p)
print(included_cvz_estimation)

reverse_image = Image.fromarray(reverse_array)
# reverse_image.show()
print()