import cv2

cap = cv2.VideoCapture("../data/test_countryroad.mp4")
read_fps = cap.get(cv2.CAP_PROP_FPS)

fps = 10
thresh = read_fps / fps
frame_counter = 0

data = {"descriptors": [], "keypoints": [], "frame": []}


def featureExtract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    #sift_image = cv2.drawKeypoints(gray, keypoints, img)
    return keypoints, descriptors


def match(d1, d2, k1, k2, f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(d1, d2)
    matches = sorted(matches, key=lambda x: x.distance)
    matched_img = cv2.drawMatches(f1, k1, f2, k2, matches[:50], f2, flags=2)
    return matched_img


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter += 1

    if frame_counter >= thresh:
        keypoints, descriptors = featureExtract(frame)
        data["descriptors"].append(descriptors)
        data["keypoints"].append(keypoints)
        data["frame"].append(frame)
        if len(data["descriptors"]) == 3:
            data["descriptors"].pop(0)
            data["keypoints"].pop(0)
            data["frame"].pop(0)
            match_img = match(data["descriptors"][0], data["descriptors"][1], data["keypoints"]
                              [0], data["keypoints"][1], data["frame"][0], data["frame"][1])
            cv2.imshow("image", match_img)
        frame_counter = 0

    if cv2.waitKey(120) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
