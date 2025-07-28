import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('../thunder.webp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply threshold
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)

# Aspect Ratio
x, y, w, h = cv2.boundingRect(cnt)
aspect_ratio = float(w) / h

# Extent
area = cv2.contourArea(cnt)
rect_area = w * h
extent = float(area) / rect_area

# Solidity
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = float(area) / hull_area

# Equivalent Diameter
equi_diameter = np.sqrt(4 * area / np.pi)

# Orientation
(x_ell, y_ell), (MA, ma), angle = cv2.fitEllipse(cnt)

# Mask and pixel points
mask = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask, [cnt], 0, 255, -1)
pixelpoints = np.transpose(np.nonzero(mask))

# Min/Max values
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray, mask=mask)

# Mean intensity
mean_val = cv2.mean(img, mask=mask)

# Extreme points
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

# ---------------- Individual Visualizations ----------------

# 1. Bounding Box
img_box = img.copy()
cv2.rectangle(img_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 2. Ellipse
img_ellipse = img.copy()
cv2.ellipse(img_ellipse, ((x_ell, y_ell), (MA, ma), angle), (0, 255, 255), 2)

# 3. Equivalent Diameter Circle
img_circle = img.copy()
cv2.circle(img_circle, (int(x_ell), int(y_ell)), int(equi_diameter / 2), (0, 255, 0), 2)

# 4. Extreme Points
img_extremes = img.copy()
cv2.circle(img_extremes, leftmost, 5, (255, 0, 0), -1)
cv2.circle(img_extremes, rightmost, 5, (0, 255, 255), -1)
cv2.circle(img_extremes, topmost, 5, (255, 0, 255), -1)
cv2.circle(img_extremes, bottommost, 5, (0, 0, 255), -1)

# 5. Mask Visualization
mask_colored = cv2.merge([mask, mask, mask])  # Convert to 3-channel for Matplotlib

# Convert BGR to RGB for plotting
img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB)
img_ellipse = cv2.cvtColor(img_ellipse, cv2.COLOR_BGR2RGB)
img_circle = cv2.cvtColor(img_circle, cv2.COLOR_BGR2RGB)
img_extremes = cv2.cvtColor(img_extremes, cv2.COLOR_BGR2RGB)
mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
thresh_rgb = cv2.cvtColor(cv2.merge([thresh]*3), cv2.COLOR_BGR2RGB)

# ---------------- Print All Properties ----------------
print("=== Contour Properties ===")
print(f"1. Aspect Ratio: {aspect_ratio:.3f}")
print(f"2. Extent: {extent:.3f}")
print(f"3. Solidity: {solidity:.3f}")
print(f"4. Equivalent Diameter: {equi_diameter:.3f}")
print(f"5. Orientation Angle: {angle:.3f}")
print(f"6. Pixel Points Count: {pixelpoints.shape[0]}")
print(f"7. Min Value: {min_val} at {min_loc}")
print(f"   Max Value: {max_val} at {max_loc}")
print(f"8. Mean BGR Color: {mean_val}")
print(f"9. Extreme Points:")
print(f"   Left: {leftmost}, Right: {rightmost}")
print(f"   Top: {topmost}, Bottom: {bottommost}")

# ---------------- Display Each One ----------------
plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.imshow(img_box)
plt.title('1. Bounding Rectangle')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_ellipse)
plt.title('2. Fit Ellipse')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_circle)
plt.title('3. Equivalent Diameter Circle')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_extremes)
plt.title('4. Extreme Points')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(mask_colored)
plt.title('5. Mask Image')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(thresh_rgb)
plt.title('6. Thresholded Image')
plt.axis('off')

plt.tight_layout()
plt.show()

