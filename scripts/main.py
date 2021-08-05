import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("../data/test_1.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.01)
ret, dst = cv2.threshold(dst, 0.06*dst.max(), 255, 0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray, np.float32(
    centroids), (5, 5), (-1, -1), criteria)
points = []
for i in range(1, len(corners)):
    points.append(corners[i])

image[dst > 0.06*dst.max()] = [0, 255, 0]
for point in points:
    plt.scatter(point[0], point[1], c="g", s=5)

# plt.imshow(image)
plt.gca().invert_yaxis()
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows
