import cv2
import joblib
import numpy as np
import core.utils as utils

img = cv2.imread("../resource/imgs/1mblue1.bmp")
clf = joblib.load("../model/mlf200-5_6_1.m")
sqs = utils.findSquare(img)
for s in sqs:
    cv2.imshow("img", s[0])
    cv2.waitKey()
    img = cv2.rectangle(img, (s[1][0], s[1][1]), (s[1][0] + s[1][2], s[1][1] + s[1][3]), (0, 255, 255), 2)
    cv2.imshow("show", img)
    cv2.waitKey()
    hog = utils.getHog(s[0])
    print(clf.predict(np.array([hog])))
    cv2.waitKey()
cv2.destroyAllWindows()
