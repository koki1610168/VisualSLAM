import numpy as np

W = 860
H = 540
focal_length_distance = 200

K = np.array([[focal_length_distance * 1/W, 0, W//2],
              [0, focal_length_distance * 1/H, H//2],
              [0, 0, 1]])

Kinv = np.linalg.inv(K)

print(Kinv)
