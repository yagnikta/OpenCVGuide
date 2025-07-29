import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the full image
target = cv2.imread('../faces.jpg')  # Full scene
target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

# Step 2: Define ROI coordinates and crop it
x1, y1, x2, y2 = 375, 175, 425, 210
roi = target[y1:y2, x1:x2]
roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

# Step 3: Convert both to HSV
hsv_target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Step 4: Histogram of ROI
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Step 5: Backprojection
back_proj = cv2.calcBackProject([hsv_target], [0], roi_hist, [0, 180], scale=1)

# Optional clean-up
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
back_proj = cv2.filter2D(back_proj, -1, disc)
_, thresh = cv2.threshold(back_proj, 50, 255, cv2.THRESH_BINARY)

# Final masked result
res = cv2.bitwise_and(target, target, mask=thresh)
res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

# Plot all images
plt.figure(figsize=(12, 8))

images = [target_rgb, roi_rgb, back_proj, thresh, res_rgb]
titles = ['Original Image', 'Cropped ROI', 'BackProjection', 'Thresholded Mask', 'Final Result']

for i in range(5):
    plt.subplot(2, 3, i + 1)
    if i in [2, 3]:  # grayscale images
        plt.imshow(images[i], cmap='gray')
    else:            # RGB images
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()