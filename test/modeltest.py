import joblib
import core.utils as utils

print("训练模型精度查看")
clf = joblib.load("../model/mlf_small_6_1.m")
feature_test, label_test = utils.read_set("../resource/testsmall")
print("精度: " + str(clf.score(feature_test, label_test)))
