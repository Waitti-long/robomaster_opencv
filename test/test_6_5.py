import cv2
import core.utils as utils
import joblib
import numpy as np

clf = joblib.load("../model/mlf2_1.m")
image = cv2.imread("../resource/res/6.5mr2.bmp")
sqs = utils.findSquare(image)
for s in sqs:
    cv2.imshow("img", s[0])
    cv2.waitKey()
    hog = utils.getHog(s[0])
    sc = clf.predict(np.array([hog]))
    if sc != 0 and sc != 5:
        # print(sc)
        img = cv2.rectangle(image, (s[1][0], s[1][1]), (s[1][0] + s[1][2], s[1][1] + s[1][3]), (0, 255, 255), 2)
        img = cv2.putText(img, str(sc), (s[1][0], s[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("show", img)
        cv2.waitKey()

cv2.destroyAllWindows()
