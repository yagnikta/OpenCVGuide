import cv2
import numpy as np

# Load the image
image = cv2.imread('../hand.jpg')  # Example with clear shapes
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Optional: Dilation to strengthen edges
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(gray, kernel, iterations=1)

cv2.imshow('Dilated', dilated)
cv2.waitKey(1)
cv2.destroyAllWindows()

# Thresholding
ret, thresh = cv2.threshold(dilated, 100, 200, cv2.THRESH_BINARY)

cv2.imshow("Threshold", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Create a blank canvas to draw colored contours
output = image.copy()

if contours:
    print(hierarchy)
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.drawContours(output, contours, i, (0,0,255), 2)
        cv2.imshow('Colored Contours', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

