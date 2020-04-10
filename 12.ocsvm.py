import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.externals import joblib
import joblib


#画precision/recall曲线函数
def plot_precision_recall_threshold(precisions,recalls,thresholds):
    plt.plot(thresholds,precisions[:-1],"b--",label="Precision")
    plt.plot(thresholds,recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    #plt.legend(locals="upperleft")
    plt.ylim([0,1])

#画ROC曲线函数
def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth = 2,label = label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel("Flase Positive Rate")
    plt.ylabel("True Positive Rate")



Positive = np.load(r"C:\Users\艾鑫坤\Desktop\data\positive.npy")
Negative = np.load(r"C:\Users\艾鑫坤\Desktop\data\negative.npy")

NegativeTestLabel = []
for i in range(len(Negative)):
    NegativeTestLabel.append(-1)

Positive = np.random.permutation(Positive)
PositiveTest = Positive[:5700]
TrainData = Positive[5700:]
PositiveTestLabel = []
for i in range(len(PositiveTest)):
    PositiveTestLabel.append(1)


#将测试数据合并，并打乱
TestData_Temp = list(PositiveTest) + list(Negative)
TestLabel_Temp = PositiveTestLabel + NegativeTestLabel
TestIndex = np.arange(len(TestLabel_Temp))
TestIndex = np.random.permutation(TestIndex)
TestData = []
TestLabel = []
for i in TestIndex:
    TestData.append(TestData_Temp[i])
    TestLabel.append(TestLabel_Temp[i])
print("数据处理完成")


#训练
clf = svm.OneClassSVM(kernel="rbf",gamma=100, nu=0.8)
clf.fit(TrainData)
joblib.dump(clf,r"C:\Users\艾鑫坤\Desktop\data\模型\ocsvmModel.pkl")
print("训练完成")


#评价
TestPredict = clf.predict(TestData)
n = 0
for i in range(len(TestPredict)):
    if TestLabel[i] == TestPredict[i]:
        n += 1
score = n/(len(TestPredict))
print("返回给定测试集和对应标签的平均准确率")
print(score)

# 交叉评估
from sklearn.model_selection import cross_val_score
score = cross_val_score(clf, TestData, TestLabel, cv=3, scoring="accuracy")
print("cross_val_score ： {}".format(score))

#precision-recall曲线
label_score = clf.decision_function(TestData)
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(TestLabel, label_score)
plot_precision_recall_threshold(precisions, recalls, thresholds)
plt.show()

#ROC曲线
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(TestLabel, label_score)
plot_roc_curve(fpr, tpr)
plt.show()

#AUC
from sklearn.metrics import roc_auc_score
print("AUC : ")
print(roc_auc_score(TestLabel, label_score))




