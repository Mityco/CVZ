import numpy as np
from PIL import Image
from cmath import phase

CVZ = np.random.normal(0, 1, size=[256, 256])
CVZ = (CVZ - CVZ.mean()) / CVZ.std()

image = Image.open("bridge.tif")
image.show()
image_array = np.asarray(image)

spectre_array = np.fft.fft2(image_array)
get_phase = np.vectorize(phase)
phase_array = get_phase(spectre_array)

abs_spectre = abs(spectre_array)
changed_abs_spectre = abs_spectre
changed_abs_spectre[128:384, 128:384] = abs_spectre[128:384, 128:384] + 100000*CVZ
changed_spectre = changed_abs_spectre * np.exp(phase_array*1j)

# spectre_image = Image.fromarray(spectre_array)
reverse_array = abs(np.fft.ifft2(changed_spectre))

with open("reverse_array.txt", "w+") as f:
    f.write(str(reverse_array))

reverse_image = Image.fromarray(reverse_array)
reverse_image.show()
print()
