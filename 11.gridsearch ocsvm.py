import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV


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
TrainDataLabel = []
for i in range(len(TrainData)):
    TrainDataLabel.append(1)
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
param_grid = {
              "gamma":[0.001, 0.01, 0.1, 1, 10, 100],
              "nu":[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
             }
print("parametes : {}".format(param_grid))
clf = svm.OneClassSVM(kernel="rbf",verbose=True)
grid_search = GridSearchCV(clf,param_grid,cv=5,scoring="accuracy")
grid_search.fit(TrainData,TrainDataLabel)
print("训练完成")


#评价
print("Best parameters : {}".format(grid_search.best_params_))
print("Test score : {}".format(grid_search.score(TestData, TestLabel)))
print("Best score on train : {}".format(grid_search.best_score_))


