import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../image.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
b, g, r = cv2.split(image_rgb)

# Normalization using normalize function, NORM_MINMAX is used for converting the pixel values between [0, 1]
# converted to float for accurate computations since uint8 can overflow or give incorrect results.

b_normalized = cv2.normalize(b.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
g_normalized = cv2.normalize(g.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
r_normalized = cv2.normalize(r.astype('float'), None, 0, 1, cv2.NORM_MINMAX)

#combines separately normalized rbg channels
normalized_image = cv2.merge((b_normalized, g_normalized, r_normalized))
print(normalized_image[:, :, 0])

plt.imshow(normalized_image)
plt.xticks([]),
plt.yticks([]),
plt.title('Normalized Image')
plt.show()