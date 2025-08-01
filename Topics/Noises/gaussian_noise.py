import cv2
import numpy as np

img = cv2.imread('../pyramid.jpeg')

blur = cv2.GaussianBlur(img, (5,5), 10)

cv2.imshow('Original Image', img)
cv2.imshow('Gaussian Blur', blur)

cv2.waitKey(0)

