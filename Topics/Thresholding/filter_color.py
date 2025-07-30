"""
cv2.VideoCapture(0) — opens the default webcam (0 = first camera).
Each frame is converted from BGR to HSV (Hue, Saturation, Value).
A color range for blue is defined.
cv2.inRange(...) creates a.jpg mask where blue pixels are white (255), others are black (0).
bitwise_and(...) applies this mask to show only the blue areas.
The result is shown in three windows:
Original frame
Binary mask
Masked output (only blue regions)

"""

import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('../../dump/b__.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for consistent display size
    frame = cv2.resize(frame, (400, 300))

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define blue range in HSV
    lower_blue = np.array([60, 35, 140])
    upper_blue = np.array([180, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Create result by masking
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert mask to 3 channels to display beside color images
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Stack all three frames horizontally
    combined = np.hstack((frame, mask_bgr, result))

    # Show the combined output
    cv2.imshow('Original | Mask | Result', combined)

    # Press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()