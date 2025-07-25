import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the input image in grayscale
img = cv2.imread('../sharp.jpg', 0)

# Creating a kernel of size 5x5
kernel = np.ones((5, 5), np.float32)

# Applying erosion and dilation
img_erosion = cv2.erode(img, kernel, iterations=2)
img_dilation = cv2.dilate(img, kernel, iterations=2)

# Plotting the images side by side using matplotlib
titles = ['Original Image', 'Erosion', 'Dilation']
images = [img, img_erosion, img_dilation]

plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
