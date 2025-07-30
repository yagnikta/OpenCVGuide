import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def readImage():

    #build image path
    root = os.getcwd()
    imgPath = os.path.join(root, 'dump/test.jpeg')

    #read image
    img = cv.imread(imgPath)

    #show image
    cv.imshow('img',img)
    cv.waitKey(0)

def writeImage():

    #build image path
    root = os.getcwd()
    imgPath = os.path.join(root, 'dump/test.jpeg')

    #read image
    img = cv.imread(imgPath)

    #write on the image
    outPath = os.path.join(root, 'dump/output.jpeg')
    cv.imwrite(outPath, img)

if __name__ == '__main__':
    readImage()
    # writeImage()