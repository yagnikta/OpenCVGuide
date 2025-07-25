"""
cv2.THRESH_BINARY: If pixel intensity is greater than the set threshold, value set to 255, else set to 0 (black).
cv2.THRESH_BINARY_INV: Inverted or Opposite case of cv2.THRESH_BINARY.
cv.THRESH_TRUNC: If pixel intensity value is greater than threshold, it is truncated to the threshold. The pixel values are set to be the same as the threshold. All other values remain the same.
cv.THRESH_TOZERO: Pixel intensity is set to 0, for all the pixels intensity, less than the threshold value.
cv.THRESH_TOZERO_INV: Inverted or Opposite case of cv2.THRESH_TOZERO.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
image1 = cv2.imread('../sharp.jpg')

# Convert to grayscale
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Apply different thresholding techniques
ret, thresh1 = cv2.threshold(img, 120, 200, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 120, 200, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 120, 200, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 120, 200, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 120, 200, cv2.THRESH_TOZERO_INV)

# Store the thresholded images and their titles
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
titles = [
    'Original Image',
    'Binary Threshold',
    'Binary Threshold Inverted',
    'Truncated Threshold',
    'Set to 0',
    'Set to 0 Inverted'
]

# Plot using matplotlib
plt.figure(figsize=(15, 6))
for i in range(6):
    plt.subplot(1, 6, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=10)
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
