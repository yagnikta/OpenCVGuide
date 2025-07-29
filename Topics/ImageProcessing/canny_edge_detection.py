import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_blur(img, ksize=(5, 5), sigma=1.4):
    return cv2.GaussianBlur(img, ksize, sigma)


def compute_gradients(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(sobel_x, sobel_y)
    magnitude = magnitude / magnitude.max() * 255
    direction = np.arctan2(sobel_y, sobel_x)
    return magnitude, direction


def non_max_suppression(magnitude, direction):
    M, N = magnitude.shape
    Z = np.zeros((M, N), dtype=np.uint8)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q, r = 255, 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                Z[i, j] = magnitude[i, j]
            else:
                Z[i, j] = 0

    return Z


def double_threshold(img, low_ratio=0.5, high_ratio=0.3):
    high = img.max() * high_ratio
    low = high * low_ratio

    res = np.zeros_like(img, dtype=np.uint8)

    strong = 255
    weak = 75

    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong


def hysteresis(img, weak, strong):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                if any(img[i + di, j + dj] == strong for di in [-1, 0, 1] for dj in [-1, 0, 1]):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


def manual_canny(image_path):
    original = cv2.imread(image_path, 0)
    blurred = gaussian_blur(original)
    grad_mag, grad_dir = compute_gradients(blurred)
    suppressed = non_max_suppression(grad_mag, grad_dir)
    thresholded, weak, strong = double_threshold(suppressed)
    final = hysteresis(thresholded.copy(), weak, strong)

    # Plotting all steps
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1), plt.imshow(original, cmap='gray'), plt.title("Original")
    plt.axis('off')

    plt.subplot(2, 3, 2), plt.imshow(blurred, cmap='gray'), plt.title("Gaussian Blur")
    plt.axis('off')

    plt.subplot(2, 3, 3), plt.imshow(grad_mag, cmap='gray'), plt.title("Gradient Magnitude")
    plt.axis('off')

    plt.subplot(2, 3, 4), plt.imshow(suppressed, cmap='gray'), plt.title("Non-Max Suppression")
    plt.axis('off')

    plt.subplot(2, 3, 5), plt.imshow(thresholded, cmap='gray'), plt.title("Double Threshold")
    plt.axis('off')

    plt.subplot(2, 3, 6), plt.imshow(final, cmap='gray'), plt.title("Final Canny Edge")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    manual_canny('../coins.jpg')  # Replace with your own image path
