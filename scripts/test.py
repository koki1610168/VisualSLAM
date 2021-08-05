import numpy as np
import cv2
import matplotlib.pyplot as plt
from extract import Extract

if __name__ == "__main__":
    cap = cv2.VideoCapture("../data/harder_challenge_video.mp4")
    frameTime = 10
    fe = Extract()

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (860, 540))
        fe.featureExtract(frame)
# result is dilated for marking the corners, not importan
# Threshold for an optimal value, it may vary depending on the image.

        cv2.imshow('dst', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()

    cv2.destroyAllWindows()
