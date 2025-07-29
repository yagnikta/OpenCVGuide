import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
If you pass cv2.CHAIN_APPROX_NONE, all the boundary points are stored. But actually do we need all the points? For eg, you found the contour of a straight line. Do you need all the points on the line to represent that line? No, we need just two end points of that line. This is what cv2.CHAIN_APPROX_SIMPLE does. It removes all redundant points and compresses the contour, thereby saving memory.
"""


# Load the image
image = cv2.imread('../coins.jpg')  # example with clear shapes
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold
ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

"""
cv2.RETR_EXTERNAL	Only outer contours (ignores inner nested contours)
cv2.RETR_LIST	All contours, but no hierarchy (just a flat list)
cv2.RETR_TREE	All contours with full hierarchy info (parent-child nesting structure)
cv2.RETR_CCOMP	Two-level hierarchy (outer and inner contours only)
"""

# Draw contours on a copy of the original
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

# Calculate the area inside the contour
area = cv2.contourArea(contours[0])
print("Area:", area)

# Calculates the perimeter of the outer contour. It is also called arc length. It can be found out using cv2.arcLength() function. Second argument specify whether shape is a closed contour (if passed True), or just a curve.
perimeter = cv2.arcLength(contours[0], True)
print("Perimeter" , perimeter)

"""
Contour Approximation:

epsilon = 0.1 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

cnt: The original shape you found using cv2.findContours().
epsilon: How much error or "wiggle" you're willing to allow. It's a percentage of the perimeter (arc length) of the shape.
True: Means the contour is closed (it loops back to the start).



"""

# Display using matplotlib
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Contours')
plt.axis('off')
plt.show()
