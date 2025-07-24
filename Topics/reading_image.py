import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image_color(path):
    # To read image from disk, we use
    # cv2.imread function, in below method,
    return cv2.imread(path, cv2.IMREAD_COLOR)

def read_image_grayscale(path):
    # Using cv2.imread() method
    # Using 0 to read image in grayscale mode
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def display_image_cv(title, img):
    # Creating GUI window to display an image on screen
    # first Parameter is windows title (should be in string format)
    # Second Parameter is image array
    cv2.imshow(title, img)
    # To hold the window on screen, we use cv2.waitKey method
    # Once it detected the close input, it will release the control
    # To the next line
    # First Parameter is for holding screen for specified milliseconds
    # It should be positive integer. If 0 pass as parameter, then it will
    # hold the screen until user closes it.
    cv2.waitKey(0)
    # It is for removing/deleting created GUI window from screen
    # and memory
    cv2.destroyAllWindows()

def display_with_matplotlib_comparison(img_bgr):
    # Converting BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Displaying image using plt.imshow() method
    plt.figure()
    plt.subplot(121)
    plt.imshow(img_bgr)
    plt.title('without converting the channels')
    plt.subplot(122)
    plt.imshow(img_rgb)
    plt.title('after converting the channels')

    # hold the window
    plt.waitforbuttonpress()
    plt.close('all')

if __name__ == "__main__":
    img_path = "image.jpeg"

    # Read and display color image with OpenCV
    img_color = read_image_color(img_path)
    display_image_cv("image", img_color)

    # Display comparison using Matplotlib (BGR vs RGB)
    display_with_matplotlib_comparison(img_color)

    # Read and display grayscale image with OpenCV
    img_gray = read_image_grayscale(img_path)
    display_image_cv("image", img_gray)
