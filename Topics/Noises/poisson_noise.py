import cv2
import numpy as np
import random

img = cv2.imread('../pyramid.jpeg')
img = img.astype(np.uint8)

noise = np.random.poisson(50, img.shape).astype(np.uint8)

output = img + noise
output = np.clip(output,0,255).astype(np.uint8)

cv2.imshow('output', output)

cv2.waitKey(0)

cv2.destroyAllWindows()