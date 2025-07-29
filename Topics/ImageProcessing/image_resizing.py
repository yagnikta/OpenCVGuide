"""
cv2.resize(): Resizes the image to new dimensions.
cv2.INTER_CUBIC: Provides high-quality enlargement.
cv2.INTER_AREA: Works best for downscaling.
"""

import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread('../image.jpeg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

scale_factor_1 = 3.0 #upsize
scale_factor_2 = 1 / 3.0 #downsizee

height, width = image_rgb.shape[:2] #take out height and width

new_height = int(height * scale_factor_1)
new_width = int(width * scale_factor_1)

zoomed_image = cv.resize(src=image_rgb,
                          dsize=(new_width, new_height),
                          interpolation=cv.INTER_CUBIC) #main function.

new_height1 = int(height * scale_factor_2)
new_width1 = int(width * scale_factor_2)

scaled_image = cv.resize(src=image_rgb,
                          dsize=(new_width1, new_height1),
                          interpolation=cv.INTER_AREA)

fig, axs = plt.subplots(1, 3, figsize=(10, 4))
fig.suptitle("Resizing") #supertitle

axs[0].imshow(image_rgb)
axs[0].set_title('Original Image Shape:' + str(image_rgb.shape)) #individual title for plots
axs[1].imshow(zoomed_image)
axs[1].set_title('Zoomed Image Shape:' + str(zoomed_image.shape))
axs[2].imshow(scaled_image)
axs[2].set_title('Scaled Image Shape:' + str(scaled_image.shape))

# removes the x-axis and y-axis marks from plot resulting in less clutter
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
fig.savefig("output.jpeg")

original_bgr = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
zoomed_bgr = cv.cvtColor(zoomed_image, cv.COLOR_RGB2BGR)
scaled_bgr = cv.cvtColor(scaled_image, cv.COLOR_RGB2BGR)

# Save images
cv.imwrite("original_image.jpeg", original_bgr)
cv.imwrite("zoomed_image.jpeg", zoomed_bgr)
cv.imwrite("scaled_image.jpeg", scaled_bgr)