import cv2
import numpy as np

# Load the image
image = cv2.imread('../test.jpg')  # Change to your image file name
original = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold to make it binary
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over each contour
for cnt in contours:

    # Draw original contour in blue
    cv2.drawContours(original, [cnt], -1, (255, 0, 0), 1)

    # Calculate perimeter
    perimeter = cv2.arcLength(cnt, True)

    # Calculate epsilon (10% of perimeter)
    epsilon = 0.02 * perimeter

    # Approximate contour
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Draw approximated contour in green
    cv2.drawContours(original, [approx], -1, (0, 255, 0), 1)


# Show result
cv2.imshow('Original vs Approximated Contours', original)
cv2.waitKey(0)
cv2.destroyAllWindows()
