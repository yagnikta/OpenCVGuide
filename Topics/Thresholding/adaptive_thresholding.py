"""
- Adaptive thresholding is the method where the threshold value is calculated for smaller regions. This leads to different threshold values for different regions with respect to the change in lighting. We use cv2.adaptiveThreshold for this.

Syntax: cv2.adaptiveThreshold(source, maxVal, adaptiveMethod, thresholdType, blocksize, constant)

Parameters:
-> source: Input Image array (Single-channel, 8-bit or floating-point)
-> maxVal: Maximum value that can be assigned to a.jpg pixel.
-> adaptiveMethod: Adaptive method decides how threshold value is calculated.

 cv2.ADAPTIVE_THRESH_MEAN_C: Threshold Value = (Mean of the neighbourhood area values - constant value). In other words, it is the mean of the blockSize×blockSize neighborhood of a.jpg point minus constant.

cv2.ADAPTIVE_THRESH_GAUSSIAN_C: Threshold Value = (Gaussian-weighted sum of the neighbourhood values - constant value). In other words, it is a.jpg weighted sum of the blockSize×blockSize neighborhood of a.jpg point minus constant.

-> thresholdType: The type of thresholding to be applied.
-> blockSize: Size of a.jpg pixel neighborhood that is used to calculate a.jpg threshold value.
-> constant: A constant value that is subtracted from the mean or weighted sum of the neighbourhood pixels.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read input image
image1 = cv2.imread('../pyramid.jpeg')

# Convert to grayscale
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Apply Adaptive Thresholding
thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 199, 5)

thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 199, 5)

# Display using matplotlib
titles = ['Original Image (Gray)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, thresh1, thresh2]

plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()