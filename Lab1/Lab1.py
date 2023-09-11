from PIL import Image
import numpy as np


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


get_plain = np.vectorize(get_plain_bit)
set_plain = np.vectorize(set_plain_bit)

image = Image.open("baboon.tif")
ornament = Image.open("ornament.tif")

red, green, blue = image.split()
arr_blue = np.array(blue)

arr_ornament = np.array(ornament)
bit_ornament = arr_ornament / 255
bit_ornament = bit_ornament.astype(np.uint8)
modified_blue = set_plain(arr_blue, bit_ornament, 3)
modified_blue = Image.fromarray(modified_blue, mode="I")
modified_blue = modified_blue.convert("L")
modified_image = Image.merge("RGB", (red, green, modified_blue))
modified_image.show()
red, green, blue = modified_image.split()
arr_blue = np.array(blue)
extracted_plain = get_plain(arr_blue, 3)
extracted_info = np.zeros((512, 512), np.uint8)
extracted_info = set_plain(extracted_info, extracted_plain, 3)
extracted_info = extracted_info * 255
extracted_image = Image.fromarray(extracted_info, mode="I")
extracted_image.show()
print()
