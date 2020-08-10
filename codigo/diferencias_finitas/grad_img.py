"""
Compute the gradient of an image.

@author:Nicolás Guarín-Zapata
@date: July 2020
"""
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    """Convert an RGB image to grayscale.

    Taken from: https://stackoverflow.com/a/12201744/3358223
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = misc.face()
img_gray = rgb2gray(img)
grad_img = np.gradient(img_gray)
mag_grad_img = np.sqrt(grad_img[0]**2 + grad_img[1]**2)
mag_grad_img *= 255/mag_grad_img.max()

#%% Visualization
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap="gray")
plt.axis("image")
plt.axis("off")


plt.subplot(1, 2, 2)
plt.imshow(mag_grad_img, cmap="viridis")
plt.axis("image")
plt.axis("off")
plt.show()

# %%
