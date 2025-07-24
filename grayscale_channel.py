import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

def grayscale():
    root = os.getcwd()
    imgPath = os.path.join(root, 'dump/test.jpeg')
    img = cv.imread(imgPath)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imshow('gray', imgGray)
    cv.waitKey(0)

def readAsGray():
    root = os.getcwd()
    imgPath = os.path.join(root, 'dump/test.jpeg')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    cv.imshow('gray', img)
    cv.waitKey(0)

if __name__ == '__main__':
    # grayscale()
    readAsGray()