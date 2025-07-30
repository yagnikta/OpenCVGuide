# organizing imports
import cv2
import numpy as np

# path to input images are specified and
# images are loaded with imread command
image1 = cv2.imread('../a.jpg')
image2 = cv2.imread('../b.jpg')

# cv2.subtract is applied over the
# image inputs with applied parameters
sub = cv2.subtract(image1, image2)

# the window showing output image
# with the subtracted image
cv2.imshow('Subtracted Image', sub)

# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

"""

dst(x, y) = src1(x, y) - src2(x, y)
But with saturation arithmetic — meaning no negative values allowed.
If a subtraction results in a negative value, OpenCV clips it to 0 instead of allowing wrap-around.


Use Case	            Description
Motion detection	    Subtract background frame from current frame to detect movement.
Image differencing	    Highlight differences between two similar images.
Masking	Subtract        mask regions to isolate foreground.
Edge sharpening	        Combine with filters to boost details by subtracting blurred version.

"""