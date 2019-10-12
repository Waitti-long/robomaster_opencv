import glob
import joblib
from sklearn import svm  # svm
from skimage import feature as ft  # hog
import numpy as np
import os
from os import listdir
from sklearn.neural_network import MLPClassifier
import time
import cv2
import core.utils as utils


def read_data(foldername):
    imgs = []
    count = 0
    filenames = glob.iglob(os.path.join(foldername, '*'))

    for filename in filenames:
        filename = filename[:filename.rfind("\\")] + "/" + filename[filename.rfind("\\") + 1:]
        src = cv2.imread(filename)
        if src is not None:
            imgs.append(src)
            count += 1

    return imgs


def cal_hog(data):
    blocks = []
    for img in data:
        imgcopy = cv2.GaussianBlur(img, (9, 9), 0)  # 高斯模糊
        imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_BGR2GRAY)  # 转灰度，找轮廓时需要
        imgcopy = cv2.adaptiveThreshold(imgcopy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 9, 2)
        imgcopy = cv2.bitwise_not(imgcopy)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        imgcopy = cv2.morphologyEx(imgcopy, cv2.MORPH_CLOSE, kernel)
        imgcopy = cv2.bitwise_not(imgcopy)

        normalised_blocks = utils.getHog(imgcopy, justHog=True)
        blocks.append(normalised_blocks)
    return blocks


if __name__ == "__main__":
    time_start = time.time()
    print("读取正面数据")
    pos_data = read_data("../resource/simple6/1")
    pos_data2 = read_data("../resource/simple6/2")
    pos_data3 = read_data("../resource/simple6/3")
    pos_data4 = read_data("../resource/simple6/4")
    pos_data5 = read_data("../resource/simple6/5")
    # print("正面数据个数：" + str(len(pos_data)))
    print("读取负面数据")
    neg_data = read_data("../resource/simple6/0")
    # print("负面数据个数：" + str(len(neg_data)))
    print("计算正面特性")
    pos_features = cal_hog(pos_data)
    pos_features2 = cal_hog(pos_data2)
    pos_features3 = cal_hog(pos_data3)
    pos_features4 = cal_hog(pos_data4)
    pos_features5 = cal_hog(pos_data5)
    print("计算反面特性")
    neg_features = cal_hog(neg_data)

    features = np.concatenate((neg_features, pos_features, pos_features2, pos_features3, pos_features4, pos_features5))
    labels = np.zeros((len(pos_data) + len(neg_data) + len(pos_data2) + len(pos_data3) + len(pos_data4) + len(pos_data5), ), dtype=np.int32)
    labels[:len(neg_data)] = 0
    labels[len(neg_data):] = 1

    print("SVM训练中")
    clf = svm.LinearSVC()
    clf.fit(features, labels)

    print("读取正面数据")
    pos_data = read_data("../resource/test/1")
    pos_data2 = read_data("../resource/test/2")
    pos_data3 = read_data("../resource/test/3")
    pos_data4 = read_data("../resource/test/4")
    pos_data5 = read_data("../resource/test/5")
    print("正面数据个数：" + str(len(pos_data)))
    print("读取负面数据")
    neg_data = read_data("../resource/test/0")
    print("负面数据个数：" + str(len(neg_data)))
    print("计算正面特性")
    pos_features = cal_hog(pos_data)
    pos_features2 = cal_hog(pos_data2)
    pos_features3 = cal_hog(pos_data3)
    pos_features4 = cal_hog(pos_data4)
    pos_features5 = cal_hog(pos_data5)
    print("计算反面特性")
    neg_features = cal_hog(neg_data)

    features = np.concatenate((neg_features, pos_features, pos_features2, pos_features3, pos_features4, pos_features5))
    labels = np.zeros(
        (len(pos_data) + len(neg_data) + len(pos_data2) + len(pos_data3) + len(pos_data4) + len(pos_data5),),
        dtype=np.int32)
    labels[:len(neg_data)] = 0
    labels[len(neg_data):] = 1

    print("保存模型中")
    joblib.dump(clf, "../model/svm6_1.m")

    print("训练结束,模型保存成功")

    print("训练模型精度查看")
    print("精度: " + str(clf.score(features, labels)))

    time_end = time.time()
    print("共计用时: " + str(time_end - time_start) + "s")
