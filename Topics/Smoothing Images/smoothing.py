# Manually define the kernel. filter2D processes the image according to the kernel provided.

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../dump/test.jpeg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.ones((5,5),np.float32)/25

dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()


# Averaging with blur / boxFilter function directly applies the kernel automatically, not flexible for other kind of filters.

img = cv2.imread('../../dump/test.jpeg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

blur = cv2.blur(img,(5,5))

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

#