import numpy as np
import cv2
import matplotlib.pyplot as plt
from extract import Extract
from mpl_toolkits.mplot3d import Axes3D
if __name__ == "__main__":
    W = 860
    H = 540
    F = 200
    K = np.array([[F * 1/W, 0, W//2],
                  [0, F * 1/H,  H//2],
                  [0, 0, 1]])

    cap = cv2.VideoCapture("../data/test_countryroad.mp4")
    frameTime = 10
    fe = Extract(K, F, W, H)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (W, H))
        # fe.featureExtract(frame)
        points, idx1, idx2 = fe.withOrb(frame)
        #points = fe.withSift(frame)
        d = []
        for p1, p2 in points:
            # print(len(p1))
            # print(len(p2))
            coor = fe.calculateIntrinsicParameters(p1)
            d.append(coor)
            # print(coor)
            a1, a2 = map(lambda x: int(round(x)), p1)
            b1, b2 = map(lambda x: int(round(x)), p2)
            if abs(a1 - b1) <= 30 and abs(a2 - b2) <= 30:
                cv2.circle(frame, (a1, a2), color=(0, 255, 0), radius=3)
                cv2.line(frame, (a1, a2), (b1, b2), color=(255, 0, 0))
        #kp, des = fe.withOrb(frame)
# result is dilated for marking the corners, not importan
# Threshold for an optimal value, it may vary depending on the image.
        # for i in points:
        #    2x, y = i.ravel()
        #frame = cv2.drawKeypoints(frame, kp, outImage=None, color=(0, 255, 0), flags=0)
        cv2.imshow('dst', frame)
        if len(points) != 0:
            break
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cap.release()

    cv2.destroyAllWindows()

    #fig = plt.figure()
    #ax = Axes3D(fig)

    for i in d:
        plt.scatter(i[0], i[1], s=1, c="green")
    plt.show()
