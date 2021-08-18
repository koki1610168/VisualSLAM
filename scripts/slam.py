import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
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
            a1, a2 = map(lambda x: int(round(x)), p1)
            b1, b2 = map(lambda x: int(round(x)), p2)
            coor = fe.calculateIntrinsicParameters(p1, a1, a2, b1, b2)
#            d.append(coor)
            if abs(a1 - b1) <= 30 and abs(a2 - b2) <= 30:
                cv2.circle(frame, (a1, a2), color=(0, 255, 0), radius=3)
                cv2.line(frame, (a1, a2), (b1, b2), color=(255, 0, 0))
        cv2.imshow('dst', frame)
#        if len(points) != 0:
#            break
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cap.release()

    cv2.destroyAllWindows()


def showPlot(points):
    fig = plt.figure()
    ax = Axes3D(fig)

    for i in points:
        ax.scatter(i[0], i[1], i[2], s=1, c="green")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    plt.show()
