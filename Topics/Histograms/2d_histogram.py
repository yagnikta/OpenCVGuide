import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image and convert to HSV
img = cv2.imread('../taj.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Calculate 2D Histogram for Hue and Saturation
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# Display using matplotlib
plt.imshow(hist, interpolation='nearest')
plt.title('2D Histogram for H & S')
plt.xlabel('Saturation')
plt.ylabel('Hue')
plt.show()
