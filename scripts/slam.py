import numpy as np
import cv2
import matplotlib.pyplot as plt
from extract import Extract

if __name__ == "__main__":
    cap = cv2.VideoCapture("../data/test_countryroad.mp4")
    frameTime = 10
    fe = Extract()

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (860, 540))
        #fe.featureExtract(frame)
        points = fe.withOrb(frame)
        #points = fe.withSift(frame)

        for p1, p2 in points:
            #print(len(p1))
            #print(len(p2))
            a1, a2 = map(lambda x: int(round(x)), p1)
            b1, b2 = map(lambda x: int(round(x)), p2)
            if abs(a1 - b1) <= 30 and abs(a2 - b2) <= 30:
                cv2.circle(frame, (a1, a2), color=(0, 255, 0), radius=3)
                cv2.line(frame, (a1, a2), (b1, b2), color=(255, 0, 0))
            
        #kp, des = fe.withOrb(frame)
# result is dilated for marking the corners, not importan
# Threshold for an optimal value, it may vary depending on the image.
        #for i in points:
        #    x, y = i.ravel()
        #frame = cv2.drawKeypoints(frame, kp, outImage=None, color=(0, 255, 0), flags=0)
        cv2.imshow('dst', frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cap.release()

    cv2.destroyAllWindows()
