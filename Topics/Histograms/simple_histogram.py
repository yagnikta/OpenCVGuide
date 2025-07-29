import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('../image.jpeg')

# Convert from BGR (OpenCV format) to RGB (matplotlib format)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create subplots: 2 rows, 2 columns
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# ------------------- [0, 0] Show Original Image -------------------
axs[0, 0].imshow(img_rgb)
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

# ------------------- [0, 1] Full Image Histogram -------------------
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img_rgb], [i], None, [255], [100, 200])
    axs[0, 1].plot(histr, color=col)
    axs[0, 1].set_xlim([0, 256])
axs[0, 1].set_title('Full Image Color Histogram')

# ------------------- [1, 0] Show Masked Region -------------------
# Create mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv2.bitwise_and(img, img, mask=mask)
masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

axs[1, 0].imshow(masked_img_rgb)
axs[1, 0].set_title('Masked Region')
axs[1, 0].axis('off')

# ------------------- [1, 1] Masked Histogram -------------------
for i, col in enumerate(color):
    histr = cv2.calcHist([masked_img_rgb], [i], mask, [255], [100, 200])
    axs[1, 1].plot(histr, color=col)
    axs[1, 1].set_xlim([0, 256])
axs[1, 1].set_title('Masked Region Histogram')

plt.tight_layout()
plt.show()
