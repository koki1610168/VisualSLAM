import numpy as np
import cv2
import matplotlib.image as mpimg

class Calibration:
    def __init__(self, path):
        self.path = path

    def calibration(self):
        nx = 9
        ny = 6

        objpoints = []
        imgpoints = []

        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        for each_img_path in self.path:
            img = mpimg.imread("../data/camera_cal/" + each_img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist
