# importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread('../taj.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply Gaussian Blur
gaussian = cv2.GaussianBlur(image, (7, 7), 0)
gaussian_rgb = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)

# Apply Median Blur
median = cv2.medianBlur(image, 5)
median_rgb = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)

# Apply Bilateral Blur
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
bilateral_rgb = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)

# Plot all images
titles = ['Original Image', 'Gaussian Blur', 'Median Blur', 'Bilateral Blur']
images = [image_rgb, gaussian_rgb, median_rgb, bilateral_rgb]

plt.imshow(median_rgb)

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()