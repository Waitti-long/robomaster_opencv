import cv2
import core.utils as utils
import joblib
import numpy as np

clf = joblib.load("../model/mlf200-5_6_1.m")
camera = cv2.VideoCapture(0)
success, frame = camera.read()
while success:
    squares = utils.findSquare(frame)
    num = {1: [], 2: [], 3: [], 4: [], 5: []}
    for square in squares:
        hog = utils.getHog(square[0])
        sc = clf.predict(np.array([hog]))
        rect = square[1]
        if sc != 0:
            num[sc].append(rect)
            frame = cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 255), 2)
            frame = cv2.putText(frame, str(sc), (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("show", frame)
        cv2.waitKey(1)
    success, frame = camera.read()

cv2.destroyAllWindows()
