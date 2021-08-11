import cv2
import numpy as np
from skimage.measure import ransac

from skimage.transform import FundamentalMatrixTransform
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
    
#ransac with skimage
    def withOrb(self, img):
        points = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(
            np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        kp = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in points]
        kp, des = self.orb.compute(img, kp)
        #img = cv2.drawKeypoints(img, kp, outImage=None, color=(0, 255, 0), flags=0)
        good = []
        p1 = []
        p2 = []
        if self.keys is not None:
            matches = self.bf.knnMatch(des, self.keys["des"], k=2)
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    p1 = kp[m.queryIdx].pt
                    p2 = self.keys["kp"][m.trainIdx].pt
                    good.append((p1, p2))
        if len(good) != 0:
            good = np.array(good)

            model, inliers = ransac((good[:, 0], good[:, 1]), 
                                    FundamentalMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=1,
                                    max_trials=100)
            good = good[inliers]
        self.keys = {"kp": kp, "des": des}
        return good
#ransac with opencv
    def withSift(self, img):
        points = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(
            np.uint8), 2000, qualityLevel=0.01, minDistance=5)
        kp = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in points]
        kp, des = self.orb.compute(img, kp)
        #img = cv2.drawKeypoints(img, kp, outImage=None, color=(0, 255, 0), flags=0)
        good = []
        p1 = []
        p2 = []
        if self.keys is not None:
            matches = self.bf.knnMatch(des, self.keys["des"], k=2)
            for m, n in matches:
                if m.distance < 0.7*n.distance:
                    p1.append(kp[m.queryIdx].pt)
                    p2.append(self.keys["kp"][m.trainIdx].pt)
                    good.append((p1, p2))
        matchesMask = []
        ok = []
        if len(good) != 0:
            p1 = np.int32(p1).reshape(-1, 1, 2)
            p2 = np.int32(p2).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(p2, p1, cv2.RANSAC, 100.0)
            matchesMask = mask.ravel().tolist()
            ok = [good[i] for i, j in enumerate(matchesMask) if j==1]
            print(ok) 
            #good = good[matchesMask]
        self.keys = {"kp": kp, "des": des}
        return ok 

