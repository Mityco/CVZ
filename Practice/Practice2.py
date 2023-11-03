from os import listdir
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

files = []
for name in listdir("Data"):
    if name.endswith(".tif"):
        files.append(f"Data\\{name}")

img = Image.open(files[0])
arr = np.asarray(img)
N1, N2 = arr.shape
print(N1, N2)

Wr = np.random.normal(0, 1, (N1, N2))
std = Wr.std()  #стандартное отклонение
mean = Wr.mean()  #среднее
Wr = (Wr - mean) / std #нормализация
new_std = Wr.std()
new_mean = Wr.mean()

alpha = 1
length = len(files)
vector0 = np.ndarray([length])
vector1 = np.ndarray([length])
vector = np.ndarray([length])

for file, i in zip(files, range(length)):
    C = np.asarray(Image.open(file))
    Cw1 = C + alpha * Wr
    Cw0 = C - alpha * Wr
    vector[i] = (1 / (N1 * N2)) * np.sum(Wr * C)
    vector0[i] = (1 / (N1 * N2)) * np.sum(Wr * Cw0)
    vector1[i] = (1 / (N1 * N2)) * np.sum(Wr * Cw1)
plt.show(Cw1, cmap="grey")

plt.hist(vector, bins=30, label="Empty images", color="red")
plt.hist(vector0, bins=30, label="Images with zero", color="green")
plt.hist(vector1, bins=30, label="Images with one", color="blue")
plt.legend()
plt.show()

tau = (new_mean + alpha) / 2
tp1 = np.sum(vector1 > tau) / length # true positive in Cw1
tp0 = np.sum(vector0 < -tau) / length # true positive in Cw0
tp = np.sum(-tau <= vector <= tau) / length
print(tp)