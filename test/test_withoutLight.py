import cv2
import core.utils as utils
import joblib
import numpy as np

clf = joblib.load("../model/mlf6_2.m")
video = cv2.VideoCapture("../resource/res/多个装甲板.avi")
success, frame = video.read()
# frame = cv2.resize(frame, (1920 // 2, 1080 // 2))
while success:
    sqs = utils.findSquaresNotUseLight(frame)
    for s in sqs:
        # cv2.imshow("img", s[0])
        # cv2.imshow("show", img)
        tmp = frame[s[1][1]:s[1][1] + s[1][2], s[1][0]:s[1][0] + s[1][2]]
        hog = utils.getHog(tmp)
        sc = clf.predict(np.array([hog]))
        if sc[0] != 0 and sc[0] != 5:
            print(sc)
            img = cv2.rectangle(frame, (s[1][0], s[1][1]), (s[1][0] + s[1][2], s[1][1] + s[1][3]), (0, 255, 255), 2)
            img = cv2.putText(img, str(sc), (s[1][0], s[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("img", frame)
    cv2.waitKey(1)
    success, frame = video.read()
    # frame = cv2.resize(frame, (1920 // 2, 1080 // 2))

cv2.destroyAllWindows()