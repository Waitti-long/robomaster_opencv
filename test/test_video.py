import cv2
import core.utils as utils
import joblib
import numpy as np

counts = np.zeros(10, dtype=np.int0)
clf = joblib.load("../model/mlfbig3.m")
clf_svm = joblib.load("../model/svm6_1.m")
video = cv2.VideoCapture("../resource/res/多个装甲板.avi")
video.set(3, 1920)
video.set(4, 1080)
success, frame = video.read()
while success:
    sqs = utils.findSquare(frame)
    for s in sqs:
        # cv2.imshow("img", s[0])
        # cv2.waitKey()
        hog = utils.getHog(s[0])
        # svm_hog = utils.getHog(s[0], justHog=True)
        # svm_sc = clf_svm.predict(np.array([svm_hog]))
        sc = clf.predict(np.array([hog]))
        counts[sc] += 1
        if sc != 0:
            # print(sc)
            frame = cv2.rectangle(frame, (s[1][0], s[1][1]), (s[1][0] + s[1][2], s[1][1] + s[1][3]), (0, 255, 255), 2)
            frame = cv2.putText(frame, str(sc), (s[1][0], s[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # else:
        #     img = cv2.rectangle(frame, (s[1][0], s[1][1]), (s[1][0] + s[1][2], s[1][1] + s[1][3]),
        #                        (0, 0, 255), 2)
        #    img = cv2.putText(img, "0", (s[1][0], s[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("show", frame)
    # print("0: ", counts[0], "1: ", counts[1], "2: ", counts[2], "3: ", counts[3], "4: ", counts[4], "5: ", counts[5])
    cv2.waitKey(1)
    success, frame = video.read()


alls = 0
for i in range(6):
    alls += counts[i]
print("counts:", alls)
for i in range(6):
    print(i, " : ", counts[i] / alls)

cv2.destroyAllWindows()
