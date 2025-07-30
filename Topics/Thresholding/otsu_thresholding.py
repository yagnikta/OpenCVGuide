"""
In Otsu Thresholding, the value of the threshold isn't chosen but is determined automatically. A bimodal image (two distinct image values) is considered. The histogram generated contains two peaks. So, a.jpg generic condition would be to choose a.jpg threshold value that lies in the middle of both the histogram peak values. We use the Traditional cv2.threshold function and use cv2.THRESH_OTSU as an extra flag.

Think of Pixel Values as a.jpg List
Your grayscale image is just a.jpg matrix of numbers from 0 (black) to 255 (white).

Suppose your image has these pixel intensities: 50, 52, 53, 48, 200, 202, 199, 205

You can already see:
Four pixel values are around 50 (dark)
Four pixel values are around 200 (light)

It’s like counting how many times each pixel value occurs in the image.

So for the pixel list above, the histogram will look like:
Near 50 → high bar
Near 200 → another high bar
In between → very few pixels

This forms two peaks: one for dark things and one for bright things.
This is called a.jpg bimodal distribution — "bi" means two peaks.

Now What Does Otsu Do?

Instead of us guessing where to cut (say threshold = 127), Otsu tries every possible value from 0 to 255 and asks:
“If I split the image at this value, how clean is the separation?”

It does this by:
Trying threshold = 1, then 2, then 3… all the way to 255.

For each threshold, it calculates how "spread out" the two groups (dark and light) are.
It picks the threshold where the dark group is tight and the light group is tight — meaning pixels are clearly grouped.
That’s the "best separation point" — and Otsu chooses that automatically.

"""

import cv2
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('../pyramid.jpeg', 0)

imt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# applying Otsu thresholding
# as an extra flag in binary
# thresholding
_, th_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Show original and thresholded images
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(th_otsu, cmap='gray')
plt.title(f'Otsu Threshold')

plt.show()