import numpy as np
import cv2
import matplotlib.pyplot as plt

# Custom Canny edge detector function
def Canny_detector(img, weak_th=None, strong_th=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 1.4)

    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mag_max = np.max(mag)
    if not weak_th: weak_th = mag_max * 0.1
    if not strong_th: strong_th = mag_max * 0.5

    height, width = img.shape

    for i_x in range(width):
        for i_y in range(height):
            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
            elif grad_ang <= 67.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
            elif grad_ang <= 112.5:
                neighb_1_x, neighb_1_y = i_x, i_y - 1
                neighb_2_x, neighb_2_y = i_x, i_y + 1
            elif grad_ang <= 157.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y - 1
            else:
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x] = 0
                    continue
            if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x] = 0

    ids = np.zeros_like(img)

    for i_x in range(width):
        for i_y in range(height):
            grad_mag = mag[i_y, i_x]
            if grad_mag < weak_th:
                mag[i_y, i_x] = 0
            elif strong_th > grad_mag >= weak_th:
                ids[i_y, i_x] = 1
            else:
                ids[i_y, i_x] = 2

    return mag

# Read image and check
frame = cv2.imread('../pyramid.jpeg')  # Update path if needed
if frame is None:
    print("Image not found.")
    exit()

# Convert for matplotlib display
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
canny_img = Canny_detector(frame)

# Plotting
fig, plots = plt.subplots(1, 2, figsize=(10, 5))
plots[0].imshow(frame_rgb)
plots[0].set_title("Original Image")
plots[0].axis("off")

plots[1].imshow(canny_img, cmap='gray')
plots[1].set_title("Edge Detection Output")
plots[1].axis("off")

plt.tight_layout()
plt.show()