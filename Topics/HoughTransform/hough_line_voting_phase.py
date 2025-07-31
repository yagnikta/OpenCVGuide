import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Load edge image using Canny
img = cv2.imread('../thunder.webp', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 50, 150)

# Get image dimensions
height, width = edges.shape

# Hough parameters
theta_resolution = 1  # degrees
rho_max = int(np.hypot(width, height))  # max rho value
rho_range = 2 * rho_max  # include negative rhos

# Accumulator array
accumulator = np.zeros((rho_range, 180), dtype=np.uint64)  # rho vs theta

# Precompute sin and cos for speed
thetas = np.deg2rad(np.arange(0, 180))
cos_t = np.cos(thetas)
sin_t = np.sin(thetas)

# Loop over edge pixels
for y in range(height):
    for x in range(width):
        if edges[y, x] == 255:
            for theta_idx in range(180):
                rho = int(round(x * cos_t[theta_idx] + y * sin_t[theta_idx]))
                rho_idx = rho + rho_max  # Shift to positive index
                accumulator[rho_idx, theta_idx] += 1

# Plot accumulator
plt.imshow(accumulator, cmap='gray', aspect='auto')
plt.title("Hough Accumulator (Rho vs Theta)")
plt.xlabel("Theta (degrees)")
plt.ylabel("Rho")
plt.show()
