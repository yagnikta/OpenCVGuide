import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('../thunder.webp')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the largest contour
cnt = max(contours, key=cv2.contourArea)

# Circle
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
imgc = img.copy()
imgc = cv2.circle(imgc, center, radius, (0, 255, 0), 2)

# Ellipse
ellipse = cv2.fitEllipse(cnt)
imge = img.copy()
imge = cv2.ellipse(imge, ellipse, (255, 0, 0), 2)

# Line
rows, cols = img.shape[:2]
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)
imgl = img.copy()
imgl = cv2.line(imgl, (cols - 1, righty), (0, lefty), (0, 0, 255), 2)

# Convert for matplotlib
imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2RGB)
imge = cv2.cvtColor(imge, cv2.COLOR_BGR2RGB)
imgl = cv2.cvtColor(imgl, cv2.COLOR_BGR2RGB)

# Show plots
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(thresh, cmap='gray')
plt.title("Thresholded Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(imgc)
plt.title("Enclosing Circle")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(imge)
plt.title("Fitted Ellipse")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(imgl)
plt.title("Fitted Line")
plt.axis("off")

plt.tight_layout()
plt.show()
