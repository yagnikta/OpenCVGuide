import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../sharp.jpg')
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

"""
Canny edge detection algorithm

The basic steps involved in this algorithm are: 

1 Noise reduction using Gaussian filter 
 
2 Gradient calculation along the horizontal and vertical axis 
 
3 Non-Maximum suppression of false edges 
 
4 Double thresholding for segregating strong and weak edges 
 
5 Edge tracking by hysteresis

"""
edges = cv2.Canny(image_rgb, 100, 700)

fig, axs = plt.subplots(1, 2, figsize=(7, 4))
axs[0].imshow(image_rgb), axs[0].set_title('Original Image')
axs[1].imshow(edges), axs[1].set_title('Image Edges')

for ax in axs:
    ax.set_xticks([]), ax.set_yticks([])

plt.tight_layout()
plt.show()