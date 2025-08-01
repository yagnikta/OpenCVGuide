import cv2
import numpy as np
import random

img = cv2.imread('../pyramid.jpeg')

output = np.zeros(img.shape, np.uint8)

prob = 0.05

thres = 1 - prob

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        rdn = random.random()
        if rdn < prob:
            output[i][j] = 0
        elif rdn > thres:
            output[i][j] = 255
        else:
            output[i][j] = img[i][j]

cv2.imshow('output', output)
cv2.waitKey(0)