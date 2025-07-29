"""
Choice of Interpolation Method for Resizing:

cv2.INTER_AREA: This is used when we need to shrink an image.
cv2.INTER_CUBIC: This is slow but more efficient.
cv2.INTER_LINEAR: This is primarily used when zooming is required. This is the default interpolation technique in OpenCV.

Syntax: cv2.resize(source, dsize, dest, fx, fy, interpolation)

Parameters:

source: Input Image array (Single-channel, 8-bit or floating-point)
dsize: Size of the output array
dest: Output array (Similar to the dimensions and type of Input image array) [optional]
fx: Scale factor along the horizontal axis  [optional]
fy: Scale factor along the vertical axis  [optional]
interpolation: One of the above interpolation methods  [optional]

"""
import os

import cv2
import matplotlib.pyplot as plt

root = os.getcwd()
path = os.path.join(root, 'image.jpeg')



image = cv2.imread(path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Loading the image

half = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)

bigger = cv2.resize(image, (1050, 1610))

stretch_near = cv2.resize(image, (780, 540),
               interpolation = cv2.INTER_LINEAR)


Titles =["Original", "Half", "Bigger", "Interpolation Nearest"]
images =[image, half, bigger, stretch_near]
count = 4

for i in range(count):
    plt.subplot(2, 2, i + 1)
    plt.title(Titles[i])
    plt.imshow(images[i])

plt.show()
plt.waitforbuttonpress()
plt.close('all')