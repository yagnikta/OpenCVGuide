import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def showPlot(imgRGB):
    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

def readAndWriteSinglePixel():
    root = os.getcwd()
    imgPath = os.path.join(root, 'test.jpeg')
    img = cv.imread(imgPath)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    eyePixel = imgRGB[312, 350]
    imgRGB[312, 350] = (0, 255, 0)

    showPlot(imgRGB)

def readAndWritePixelRegion():
    root = os.getcwd()
    imgPath = os.path.join(root, 'test.jpeg')
    img = cv.imread(imgPath)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    headRegion = imgRGB[75:140, 335:400]

    dy = 140 - 75
    dx = 400 - 335

    startY = 75
    startX = 400

    imgRGB[startY:startY + dy, startX:startX + dx] = headRegion

    startY = 75
    startX = 280

    imgRGB[startY:startY + dy, startX:startX + dx] = headRegion

    showPlot(imgRGB)

if __name__ == '__main__':
    # readAndWriteSinglePixel()
    readAndWritePixelRegion()