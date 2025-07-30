# Python program to illustrate
# arithmetic operation of
# addition of two images

# organizing imports
import cv2
import numpy as np

# path to input images are specified and
# images are loaded with imread command
image1 = cv2.imread('../a.jpg')
image2 = cv2.imread('../b.jpg')

# cv2.addWeighted is applied over the
# image inputs with applied parameters
weightedSum = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)

# the window showing output image
# with the weighted sum
cv2.imshow('Weighted Image', weightedSum)

# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

"""

dst(x,y)=src1(x,y)×α+src2(x,y)×β+γ

This formula is applied element-wise (pixel-by-pixel) for each channel (R, G, B) in both images.

Parameter	Meaning
src1	    First input image (same size and type as src2)
alpha	    Weight of the first image (src1)
src2	    Second input image
beta	    Weight of the second image (src2)
gamma	    Scalar added to the sum (brightness offset). gamma is a constant added to each resulting pixel — same value across the entire image.

🔍 Internals (Step-by-Step Execution):
For every pixel (x,y) in the image:

* Multiply the pixel value from src1 by alpha.
* Multiply the pixel value from src2 by beta.
* Add the two results.
* Add the constant gamma (if any).
* Clip the result to the range [0,255] (to stay within valid 8-bit image limits).
    - Values < 0 become 0
    - Values > 255 become 255
    - Values within range stay unchanged
    result = np.clip(result, 0, 255)
* Round the result to the nearest integer.

"""