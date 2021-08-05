import cv2
import numpy as np


class Extract:
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        self.p = {"x": [], "y": []}

    def featureExtract(self, img):
        points = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(
            np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        points = np.int0(points)
        for i in points:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            self.p["x"].append(x)
            self.p["y"].append(y)
