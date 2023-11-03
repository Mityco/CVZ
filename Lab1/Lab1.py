from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage.util import random_noise


def get_plain_bit(num, plain_num):
    filter = 1 << plain_num
    result = num & filter
    return result >> plain_num


def set_plain_bit(num, ornament_num, plain_num):
    ornament_one = ornament_num << plain_num
    ornament_zero = 255 - (1 << plain_num)
    if ornament_num == 1:
        result = num | ornament_one
    else:
        result = num & ornament_zero
    return result


sigma = 8

get_plain = np.vectorize(get_plain_bit)
set_plain = np.vectorize(set_plain_bit)

image = Image.open("baboon.tif")
ornament = Image.open("ornament.tif")

red, green, blue = image.split()
arr_blue = np.array(blue)

fig = plt.figure()

# Task 1
arr_ornament = np.array(ornament)
bit_ornament = arr_ornament / 255
bit_ornament = bit_ornament.astype(np.uint8)
modified_blue = set_plain(arr_blue, bit_ornament, 3)
modified_blue = Image.fromarray(modified_blue, mode="I")
modified_blue = modified_blue.convert("L")
modified_image = Image.merge("RGB", (red, green, modified_blue))
fig.add_subplot(221)
plt.title('Исходное изображение')
plt.imshow(image, cmap='RGB')
fig.add_subplot(221)
plt.title('Измененное изображение')
plt.imshow(modified_image, cmap='RGB')
fig.add_subplot(223)
plt.title('4ая плоскость')
plt.imshow(get_plain(), cmap='RGB')
fig.add_subplot(224)
plt.title('4ая плоскость после изменения')
plt.imshow(modified_image, cmap='RGB')
modified_image.show()



# Task 2
red, green, blue = modified_image.split()
arr_blue = np.array(blue)
extracted_plain = get_plain(arr_blue, 3)
extracted_info = np.zeros((512, 512), np.uint8)
extracted_info = set_plain(extracted_info, extracted_plain, 3)
extracted_info = extracted_info * 255
extracted_image = Image.fromarray(extracted_info, mode="I")
extracted_image.show()

# Task 3
red, green, blue = image.split()
arr_green = np.array(green)
temp = np.trunc(arr_green / (2 * sigma)) * 2 * sigma + bit_ornament * sigma
white_noise = random_noise(np.full((512, 512), 0.5)) * 7
temp += white_noise
temp = temp.astype(int)
modified_green = Image.fromarray(temp, mode="I")
modified_green = modified_green.convert("L")
new_image = Image.merge("RGB", (red, modified_green, blue))
new_image.show()

# Task 4
modified_green = np.array(modified_green)

restored_ornament = np.trunc((modified_green - np.trunc(arr_green / (2 * sigma)) * 2 * sigma - white_noise) / 7) * 255
restored_ornament = restored_ornament.astype(int)
restored_ornament = Image.fromarray(restored_ornament, mode="I")
restored_ornament.show()
print()
