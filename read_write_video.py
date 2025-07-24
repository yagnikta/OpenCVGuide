import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def readVideoFromWebCam():
    cap = cv.VideoCapture(0, cv.CAP_V4L2)

    if not cap.isOpened():
        exit()

    while True:
        ret, frame = cap.read()

        if ret:
            cv.imshow('Webcam', frame)

        if cv.waitKey(1) == ord('x'):
            break

    cap.release()
    cv.destroyAllWindows()

def readVideoFromFile():
    root = os.getcwd()
    vidPath = os.path.join(root, 'dump/test.mp4')
    cap = cv.VideoCapture(vidPath)

    while cap.isOpened():
        ret, frame = cap.read()
        cv.imshow('video', frame)
        delay = int(1000/60)
        if cv.waitKey(delay) == ord('x'):
            break

def writeVideoToFile():
    cap = cv.VideoCapture(0, cv.CAP_V4L2)

    # Use mp4v for .mp4 output
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    root = os.getcwd()
    outPath = os.path.join(root, 'dump/output.mp4')

    out = cv.VideoWriter(outPath, fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            out.write(frame)
            cv.imshow('Recording...', frame)  # Optional: show preview

        if cv.waitKey(1) == ord('x'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    # readVideoFromWebCam()
    # readVideoFromFile()
    writeVideoToFile()