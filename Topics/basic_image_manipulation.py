# Importing the OpenCV library
import os
import cv2

def read_image():
    # Reading the image using imread() function
    root = os.getcwd()
    imgPath = os.path.join(root, 'image.jpeg')
    image = cv2.imread(imgPath)
    return image

def print_image_dimensions(image):
    # Extracting the height and width of an image
    h, w = image.shape[:2]

    # Displaying the height and width
    print("Height = {}, Width = {}".format(h, w))
    return h, w

def print_pixel_values(image):
    # Extracting RGB values.
    # Here we have randomly chosen a.jpg pixel
    # by passing in 100, 100 for height and width.
    (B, G, R) = image[100, 100]

    # Displaying the pixel values
    print("R = {}, G = {}, B = {}".format(R, G, B))

    # We can also pass the channel to extract
    # the value for a.jpg specific channel
    B = image[100, 100, 0]
    print("B = {}".format(B))

def show_image(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)

def show_roi(image):
    # We will calculate the region of interest
    # by slicing the pixels of the image
    roi = image[100 : 500, 200 : 700]
    show_image("ROI", roi)

def resize_fixed(image):
    # resize() function takes 2 parameters,
    # the image and the dimensions
    resize = cv2.resize(image, (1000, 500))
    show_image("Resized Image", resize)

def resize_aspect_ratio(image, w, h):
    # Calculating the ratio
    ratio = 1000 / w

    # Creating a.jpg tuple containing width and height
    dim = (1000, int(h * ratio))

    # Resizing the image
    resize_aspect = cv2.resize(image, dim)
    show_image("Resized Image", resize_aspect)

def draw_rectangle(image):
    # We are copying the original image,
    # as it is an in-place operation.
    output = image.copy()

    # Using the rectangle() function to create a.jpg rectangle.
    rectangle = cv2.rectangle(output, (100, 50),
                              (200, 100), (0, 0, 0), 2)

    show_image("Rectangle Image", rectangle)

def draw_text(image):
    # Copying the original image
    output = image.copy()

    # Adding the text using putText() function
    text = cv2.putText(output, 'Vi and Vander', (200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (1, 1, 1), 2)

    show_image("Text Image", text)

if __name__ == '__main__':
    image = read_image()
    h, w = print_image_dimensions(image)
    print_pixel_values(image)
    show_roi(image)
    resize_fixed(image)
    resize_aspect_ratio(image, w, h)
    draw_rectangle(image)
    draw_text(image)