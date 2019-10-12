import core.utils as utils
import cv2

img = cv2.imread("../resource/simple/5/1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("1", img)
cv2.waitKey()
img = cv2.GaussianBlur(img, (3, 3), 0)
cv2.imshow("1", img)
cv2.waitKey()
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 9, 2)
cv2.imshow("1", img)
cv2.waitKey()
img = cv2.resize(img, (64, 128))
cv2.imshow("1", img)
cv2.waitKey()
