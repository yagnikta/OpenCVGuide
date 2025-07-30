"""
cv2.dilate(): Expands object boundaries.
cv2.erode(): Shrinks object boundaries.

cv2.morphologyEx() with cv2.MORPH_OPEN: Removes small noise.
cv2.morphologyEx() with cv2.MORPH_CLOSE: Fills small holes.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Ganeshji.webp')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = np.ones((3, 3), np.uint8)

dilated = cv2.dilate(image_gray, kernel, iterations=2)
eroded = cv2.erode(image_gray, kernel, iterations=2)
opening = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel, iterations=3)
closing = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel, iterations=3)

fig, axs = plt.subplots(2, 2, figsize=(7, 7))
axs[0, 0].imshow(dilated, cmap='Greys'), axs[0, 0].set_title('Dilated Image')
axs[0, 1].imshow(eroded, cmap='Greys'), axs[0, 1].set_title('Eroded Image')
axs[1, 0].imshow(opening, cmap='Greys'), axs[1, 0].set_title('Opening')
axs[1, 1].imshow(closing, cmap='Greys'), axs[1, 1].set_title('Closing')

for ax in axs.flatten():
    ax.set_xticks([]), ax.set_yticks([])

plt.tight_layout()
plt.show()

