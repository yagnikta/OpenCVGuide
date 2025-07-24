import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def showPlot(imgRGB):
    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

def pureColors():
    zeros = np.zeros((100,100))
    ones = np.ones((100,100))
    bImg = cv.merge((zeros, zeros, 255 * ones))
    gImg = cv.merge((zeros, 255*ones, zeros))
    rImg = cv.merge((255*ones, zeros, zeros))
    wImg = cv.merge((255 * ones, 255 * ones, 255 * ones))
    blImg = cv.merge((zeros, zeros, zeros))


    # the number in the subplot is showing that in the plot create the n*m grid of plots for first two digits n and m in number and assign which plot to show in which number from 1 based indexing
    plt.figure()
    plt.subplot(231)
    plt.imshow(bImg)
    plt.title('blue')
    plt.subplot(232)
    plt.imshow(gImg)
    plt.title('green')
    plt.subplot(233)
    plt.imshow(rImg)
    plt.title('red')
    plt.subplot(234)
    plt.imshow(wImg)
    plt.title('white')
    plt.subplot(235)
    plt.imshow(blImg)
    plt.title('black')

    plt.show()


def bgrChannelGrayScale():
    root = os.getcwd()
    imgPath = os.path.join(root, 'dump/test.jpeg')
    img = cv.imread(imgPath)
    b,g,r = cv.split(img)

    plt.figure()
    plt.subplot(131)
    plt.imshow(b, cmap='gray')
    plt.title('b')
    plt.subplot(132)
    plt.imshow(g, cmap='gray')
    plt.title('g')
    plt.subplot(133)
    plt.imshow(r, cmap='gray')
    plt.title('r')

    plt.show()

def bgrChannelColorScale():
    root = os.getcwd()
    imgPath = os.path.join(root, 'dump/test.jpeg')
    img = cv.imread(imgPath)
    b,g,r = cv.split(img)

    zeros = np.zeros_like(b)

    bImg = cv.merge((zeros,zeros,b))
    gImg = cv.merge((zeros, g, zeros))
    rImg = cv.merge((r,zeros, zeros))

    plt.figure()
    plt.subplot(131)
    plt.imshow(bImg)
    plt.title('b')
    plt.subplot(132)
    plt.imshow(gImg)
    plt.title('g')
    plt.subplot(133)
    plt.imshow(rImg)
    plt.title('r')

    plt.show()

if __name__ == '__main__':
    # pureColors()
    # bgrChannelGrayScale()
    bgrChannelColorScale()