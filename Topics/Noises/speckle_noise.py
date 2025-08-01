import random
import cv2
import numpy as np

img = cv2.imread('../pyramid.jpeg')

prob = 0.07
output = np.zeros(img.shape, np.uint8)
thres = 1 - prob

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        rdn = random.random()
        if rdn < prob:
            output[i][j] = 128
            for k in range(5):
                output[i-k][j-k] = 128 + 10*rdn
        else:
            output[i][j] = img[i][j]

cv2.imshow('output', output)

cv2.waitKey(0)

cv2.destroyAllWindows()