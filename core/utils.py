import numpy as np
import cv2
from sklearn import svm  # svm
import joblib
import glob
import os
from sklearn import svm  # svm
from skimage import feature as ft  # hog
from skimage.measure import compare_ssim
import pickle  # svm save
import cv2
import time


def test(img, name="show"):
    cv2.imshow(name, img)
    cv2.waitKey()


def getLight(img):
    res = np.mean(img.ravel())
    return res


def read_files(folder_name, label, justHog=False):
    img_s = []
    filenames = glob.iglob(os.path.join(folder_name, '*'))

    for filename in filenames:
        src = cv2.imread(filename)
        if src is not None:
            src = getHog(src, justHog=justHog)
            img_s.append(src)

    label_s = np.zeros((len(img_s),), dtype=np.int32)
    label_s[:] = label
    return img_s, label_s


def read_set(folder_name):
    feature_set = []
    label_set = []
    filenames = glob.iglob(os.path.join(folder_name, '*'))
    for filename in filenames:
        img_s, label_s = read_files(filename, filename[filename.rfind("\\") + 1:])
        print("有效数据个数:", len(img_s), "标签: ", label_s[0] if label_s[0] is not None else "error")
        if len(feature_set) == 0:
            feature_set = img_s
        else:
            feature_set = np.concatenate((feature_set, img_s))
        if len(label_set) == 0:
            label_set = label_s
        else:
            label_set = np.concatenate((label_set, label_s))
    return feature_set, label_set


def getHog(img, justHog=False, gauss=9):
    if not justHog:
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img1 = cv2.GaussianBlur(img1, (gauss, gauss), 0)
        img1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 9, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img1 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
        img1 = cv2.resize(img1, (64, 128))
        res = ft.hog(img1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8),
                     block_norm='L2-Hys', visualize=False)
        res = res.ravel()
        return res
    else:
        # img1 = cv2.GaussianBlur(img, (9,  9), 0)
        img1 = cv2.resize(img, (64, 128))
        res = ft.hog(img1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8),
                     block_norm='L2-Hys', visualize=False)
        res = res.ravel()
        return res


def calNumLocation2(rec1, rec2, img):
    if rec1[0] > rec2[0]:
        rec1, rec2 = rec2, rec1
    # (x, y, w, h)
    w = rec2[0] - rec1[0] + rec2[2]
    h = w + rec1[3] + rec2[3]
    x = rec1[0]
    y = rec1[1] + rec1[3] // 2 - h // 2
    # cv2.line(img, (x, y), (x + w, y + h), (255, 255, 255))
    # test(img)
    rect = (x, y, w, h)
    return rect


def findNum(img, rec1, rec2):
    rec = max(rec1, rec2, key=lambda z: z[2] * z[3])
    aver = getLight(img)
    tmp = cv2.GaussianBlur(img, (9, 9), 0)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    # 20
    # tmp = cv2.adaptiveThreshold(tmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                          cv2.THRESH_BINARY, 9, 2)
    ret, tmp = cv2.threshold(tmp, aver, 255, 0)
    squares = []

    contours, hier = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        square = img[y:y + h, x:x + w]
        if 100 < square.size and h / w > 1:
            squares.append((square, cv2.contourArea(c), (x, y, w, h)))

    squares.sort(key=lambda z: z[1], reverse=True)

    if len(squares) > 0:
        if squares[0][1] < rec[2] * rec[3] * 11:
            return squares[0][0], squares[0][2]
    return None, None


def isParallel(rec1, rec2, error=25, error2=0.3, error3=0.9):
    # (x, y, w, h)
    rec = max(rec1, rec2, key=lambda z: z[2] * z[3])
    center1 = rec1[1] + rec1[3] // 2
    center2 = rec2[1] + rec2[3] // 2
    interval = abs(rec1[0] - rec2[0])
    if interval == 0:
        return False
    if abs(center1 - center2) < error and error2 < abs(rec[3] / interval) < error3:
        return True
    return False


def isColor(img, error=80):
    result = []
    for i in range(3):
        result.append((i, np.mean(img[:, :, i])))
    result.sort(key=lambda z: z[1], reverse=True)
    if np.abs(255 - result[0][1]) < error:
        return True
    return False


def findSquare(img, hasGauss=False):
    imgcopy = img
    if not hasGauss:
        imgcopy = cv2.GaussianBlur(img, (15, 15), 0)  # 高斯模糊
    imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_BGR2GRAY)  # 转灰度，找轮廓时需要
    ret, imgcopy = cv2.threshold(imgcopy, 85, 255, 0)  # 二值化

    # test(imgcopy)

    contours, hier = cv2.findContours(imgcopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(imgcopy, (x, y), (x + w, y + h), (0, 0, 0), 2)
        tmp = img[y: y + h, x: x + w]
        if isColor(tmp):
            # cv2.rectangle(img, (x, y), (x + w, y + h),
            #               (255, 0, 0), 2)
            squares.append((x, y, w, h))

    pairList = []

    length = len(squares)

    for i in range(length):
        for j in range(i, length):
            sq1 = squares[i]
            sq2 = squares[j]
            if isParallel(sq1, sq2):
                pairList.append((sq1, sq2))
                cv2.rectangle(img, (sq1[0], sq1[1]), (sq1[0] + sq1[2], sq1[1] + sq1[3]), (255, 255, 0), 2)
                cv2.rectangle(img, (sq2[0], sq2[1]), (sq2[0] + sq2[2], sq2[1] + sq2[3]), (255, 255, 0), 2)

    numsLocations = []

    for pair in pairList:
        location = calNumLocation2(pair[0], pair[1], img)
        x, y, w, h = location
        x = max(0, x)
        y = max(0, y)
        w = np.abs(w)
        h = np.abs(h)
        # square = img[y: y + h, x: x + w]
        # test(square)
        numsLocations.append((x, y, w, h, pair[0], pair[1]))

    nums = []

    for num in numsLocations:
        square = img[num[1]:num[1] + num[3], num[0]:num[0] + num[2]]
        # test(square, "fuck")
        if square.size > 0:
            img1, loc = findNum(square, num[4], num[5])
            if img1 is not None and not isColor(img1, error=100):
                nums.append((img1, (num[0] + loc[0], num[1] + loc[1], loc[2], loc[3]), square.size))

    return nums


def isIntersect(rec, big):
    p1 = (max(rec[0], big[0]), max(rec[1], big[1]))
    p2 = (min(rec[0] + rec[2], big[0] + big[2]), min(rec[1] + rec[3], big[1] + big[3]))
    return p1[0] < p2[0] and p1[1] < p2[1]


def isContainedLight(rec, light_s):
    for light in light_s:
        if isIntersect(rec, light):
            return True
    return False


def findSquaresNotUseLight(img):
    imgcopy = cv2.GaussianBlur(img, (9, 9), 0)  # 高斯模糊
    imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_BGR2GRAY)  # 转灰度，找轮廓时需要

    lights = []
    ret, img_light = cv2.threshold(imgcopy, 95, 255, 0)  # 二值化,找灯条

    cons, hier = cv2.findContours(img_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cons:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 400:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 4)
            lights.append((x, y, w, h))

    imgcopy = cv2.adaptiveThreshold(imgcopy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 9, 2)
    imgcopy = cv2.bitwise_not(imgcopy)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    imgcopy = cv2.morphologyEx(imgcopy, cv2.MORPH_CLOSE, kernel)

    imgcopy = cv2.bitwise_not(imgcopy)

    contours, hier = cv2.findContours(imgcopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    clf_svm = joblib.load("../model/simple4_2.m")

    squares = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # print(w * h)
        i = imgcopy[y:y + h, x:x + w]

        if w * h > 100 and isContainedLight((x - w, y - h, w * 2, h * 2), lights):
            # cv2.imshow("er", i)
            # cv2.waitKey()
            hog = getHog(img, justHog=True)
            # print(clf_svm.predict(np.array([hog])))
            if clf_svm.predict(np.array([hog]))[0] == 1:
                squares.append((i, (x, y, w, h)))

    return squares


def cal_angel(point3d):
    horizontal = -1 * np.arctan(point3d[0] / point3d[2])
    vertical = -1 * np.arctan(point3d[1] / point3d[2])
    return horizontal, vertical


def pnp(points_2d, points_3d=None):
    if points_3d is None:
        points_3d = np.float32([[0, 0, 0], [135, 0, 0], [135, 125, 0], [0, 125, 0]])
    camera_marix = np.float64([[970.76885986, 0., 607.01284774],
                               [0., 969.5692749, 355.37105518],
                               [0., 0., 1.]])
    dist = np.float64([-0.17047018, 0.3970236, 0.00136597, -0.00702596, -0.37291715])
    retval, rvec, tvec, fuck = cv2.solvePnPRansac(points_3d, points_2d, camera_marix, dist)
    return cal_angel(tvec)
