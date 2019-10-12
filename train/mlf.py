import glob
import joblib
import numpy as np
import os
from os import listdir
from sklearn.neural_network import MLPClassifier
import time
import cv2
import core.utils as utils


if __name__ == "__main__":
    time_start = time.time()

    features, labels = utils.read_set("../resource/simple6_1")

    clf = MLPClassifier(hidden_layer_sizes=(200, 200, 200, 200, 200),
                        activation='relu',
                        solver='adam',
                        learning_rate_init=0.0001,
                        max_iter=2000,
                        verbose=True,
                        )

    print("MLP 训练中")
    clf.fit(features, labels)

    print("保存模型中")
    joblib.dump(clf, "../model/mlf200-5_6_2.m")

    time_end = time.time()
    print('截至MLP模型保存完毕共计用时: ', time_end - time_start, 's ', (time_end - time_start) / 60, 'min')

    print("训练模型精度查看")
    feature_test, label_test = utils.read_set("../resource/test")
    print("精度: " + str(clf.score(feature_test, label_test)))

    print("结束")
    time_end = time.time()
    print('共计用时: ', time_end - time_start, 's')

    exit()
