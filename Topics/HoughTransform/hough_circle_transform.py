import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load image and detect edges
img = cv2.imread("../coins.jpg", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 100, 200)

# Step 2: Create a 3D accumulator (a, b, r)
height, width = edges.shape
r_min, r_max = 10, 50
accumulator = np.zeros((height, width, r_max - r_min), dtype=np.uint64)

# Step 3: Loop through edge points
edge_points = np.argwhere(edges > 0)
theta = np.deg2rad(np.arange(0, 360))  # 360 directions

for x, y in edge_points:
    for r_idx, r in enumerate(range(r_min, r_max)):
        for angle in theta:
            a = int(x - r * np.cos(angle))
            b = int(y - r * np.sin(angle))
            if 0 <= a < height and 0 <= b < width:
                accumulator[a, b, r_idx] += 1

# Step 4: Find peaks in the accumulator (threshold can vary)
threshold = 120
detected_circles = np.argwhere(accumulator > threshold)

# Step 5: Draw detected circles
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for a, b, r_idx in detected_circles:
    radius = r_idx + r_min
    cv2.circle(output, (b, a), radius, (0, 255, 0), 1)

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Circles")
plt.axis('off')
plt.show()