import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../image.jpeg')
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#center point of the image.
center = (image_rgb.shape[1] // 2, image_rgb.shape[0] // 2)

angle = -30
scale = 1

#give center, angle and scale of the image; angle negative will result in rotating the image anti-clock wise.
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

#rotates the image. source image, rotated-matrix and width/height
rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (img.shape[1], img.shape[0]))


fig, axs = plt.subplots(1, 2, figsize=(7, 4))

axs[0].imshow(image_rgb)
axs[0].set_title('Original Image')
axs[1].imshow(rotated_image)
axs[1].set_title('Image Rotation')
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout() #adjusts the image layout and
plt.show()

