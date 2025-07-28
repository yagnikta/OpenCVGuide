import cv2
import numpy as np

# Load image
img = cv2.imread('../test.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to binarize
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)

# Simplify the contour
epsilon = 0.01 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

# Check convexity
print("Original Contour Convex:", cv2.isContourConvex(cnt))
print("Approximated Contour Convex:", cv2.isContourConvex(approx))

# Draw
cv2.drawContours(img, [cnt], -1, (255, 0, 0), 2)
cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

cv2.imshow("Contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
