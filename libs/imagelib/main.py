import matplotlib.pyplot as plt
import numpy as np

from imagelib import Image

size = 64
x, y, z = np.meshgrid(
    np.linspace(0, 1, size),
    np.linspace(0, 1, size),
    np.linspace(0, 1, size),
    indexing="ij",
)
array = (
    np.sin(8 * np.pi * x) * np.cos(8 * np.pi * y) + 0.1 * np.random.rand(size, size)
) * x


image = Image(array)
image_avg = image.normalize_moving_average(ax=0, window_size=32, eps=5e-3)

image.save("test.hdf5")
print(Image.load("test.hdf5", indices=(slice(0, 5), slice(None))))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image.array.T[0], extent=image.extent_imshow, origin="lower")
axes[0].set_title("Original Image")
axes[1].imshow(image_avg.array.T[0], extent=image_avg.extent_imshow, origin="lower")
axes[1].set_title("Moving Average (ax=0, window_size=32)")


plt.tight_layout()
plt.show()
