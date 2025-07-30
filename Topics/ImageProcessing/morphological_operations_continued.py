import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an image (grayscale)
image = cv2.imread('../images.jpeg', 0)

# Define a.jpg kernel (structuring element)
kernel = np.ones((5, 5), np.uint8)

# Basic Morphological Operations
erosion = cv2.erode(image, kernel, iterations=1)
dilation = cv2.dilate(image, kernel, iterations=1)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Advanced Morphological Operations
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

# Titles and images list
titles = [
    'Original Image', 'Erosion', 'Dilation', 'Opening', 'Closing',
    'Gradient', 'Top Hat', 'Black Hat'
]
images = [
    image, erosion, dilation, opening, closing,
    gradient, tophat, blackhat
]

# Plotting all images
plt.figure(figsize=(15, 10))
for i in range(len(images)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
