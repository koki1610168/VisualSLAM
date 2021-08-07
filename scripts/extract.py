import cv2
import numpy as np


#Extract points from an image
class Extract:
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        self.p = {"x": [], "y": []}
        self.keys = None
        self.orb = cv2.ORB_create()

    #Extract option 1
    def featureExtract(self, img):
        points = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(
            np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        points = np.int0(points)
        return points
        """
        for i in points:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            self.p["x"].append(x)
            self.p["y"].append(y)
        """
            
    #Extract option 2
    def withOrb(self, img):
        points = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(
            np.uint8), 2000, qualityLevel=0.01, minDistance=5)
        kp = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in points]
        kp, des = self.orb.compute(img, kp)
        #img = cv2.drawKeypoints(img, kp, outImage=None, color=(0, 255, 0), flags=0)
        good = []
        if self.keys is not None:
            matches = self.bf.knnMatch(des, self.keys["des"], k=2)
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    good.append((kp[m.queryIdx], self.keys["kp"][m.trainIdx]))
        self.keys = {"kp": kp, "des": des}
        return good
    """
    def match(self, kp, des):
        matches = None
        good = []
        if self.keys is not None:
            matches = self.bf.knnMatch(des, self.keys["des"], k=2)
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    good.append((kp[m.queryIdx].pt, self.keys["kp"][m.queryIdx].pt))
    """
