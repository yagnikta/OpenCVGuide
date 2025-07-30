import cv2
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def perpendicular_distance(point, start, end):
    if np.all(start == end):
        return euclidean_distance(point, start)
    return np.abs(np.cross(end - start, start - point)) / np.linalg.norm(end - start)

def ramer_douglas_peucker(points, epsilon):
    if len(points) < 3:
        return points

    start, end = points[0], points[-1]
    max_dist = 0
    index = 0

    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], start, end)
        if dist > max_dist:
            index = i
            max_dist = dist

    if max_dist > epsilon:
        left = ramer_douglas_peucker(points[:index + 1], epsilon)
        right = ramer_douglas_peucker(points[index:], epsilon)
        return np.vstack((left[:-1], right))
    else:
        return np.array([start, end])

# Load the image
image = cv2.imread('../hand.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Select the largest contour
cnt = max(contours, key=cv2.contourArea)

# Flatten the contour for our custom RDP function
cnt_flat = cnt[:, 0, :]  # shape (N, 2)

# Apply our custom RDP implementation
epsilon = 50.0
approx = ramer_douglas_peucker(cnt_flat, epsilon)

# Draw original and approximated contours
output = image.copy()
cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
cv2.polylines(output, [approx.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Green: Original Contour, Red: Approximated Contour")
plt.axis('off')
plt.tight_layout()
plt.show()