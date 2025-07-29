import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def histogramEqual():
    img_color = cv.imread('../image.jpeg')
    img = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)  # ✅ Convert to grayscale

    # Original Histogram and CDF
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdfNorm = cdf * float(hist.max()) / cdf.max()

    plt.figure(figsize=(12, 6))

    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdfNorm, color='b')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('# of Pixels')
    plt.title('Original Histogram')

    # Global Histogram Equalization
    equImg = cv.equalizeHist(img)
    equhist = cv.calcHist([equImg], [0], None, [256], [0, 256])
    equcdf = equhist.cumsum()
    equcdfNorm = equcdf * float(equhist.max()) / equcdf.max()

    plt.subplot(232)
    plt.imshow(equImg, cmap='gray')
    plt.title('Equalized Image')

    plt.subplot(235)
    plt.plot(equhist)
    plt.plot(equcdfNorm, color='b')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('# of Pixels')
    plt.title('Equalized Histogram')

    # CLAHE
    claheObj = cv.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
    claheImg = claheObj.apply(img)
    clahehist = cv.calcHist([claheImg], [0], None, [256], [0, 256])
    clahecdf = clahehist.cumsum()
    clahecdfNorm = clahecdf * float(clahehist.max()) / clahecdf.max()

    plt.subplot(233)
    plt.imshow(claheImg, cmap='gray')
    plt.title('CLAHE Image')

    plt.subplot(236)
    plt.plot(clahehist)
    plt.plot(clahecdfNorm, color='b')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('# of Pixels')
    plt.title('CLAHE Histogram')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    histogramEqual()