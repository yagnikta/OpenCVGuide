import cv2
import numpy as np

# Load image and preprocess
img = cv2.imread('../coins.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)

# Simplify the contour to reduce noise
epsilon = 0.01 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

# Get convex hull indices from simplified contour
hull = cv2.convexHull(approx, returnPoints=False)

# Convexity defects using original contour and simplified hull
defects = cv2.convexityDefects(approx, hull)

# Draw simplified contour
cv2.drawContours(img, [approx], -1, (255, 0, 0), 1)

# Draw convexity defect points
if defects is not None:
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        far = tuple(approx[f][0])
        cv2.circle(img, far, 5, (0, 0, 255), -1)

# Display
cv2.imshow('Refined Convexity Defects', img)
cv2.waitKey(0)
cv2.destroyAllWindows()