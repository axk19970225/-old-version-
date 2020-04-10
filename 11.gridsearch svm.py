import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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

np.random.seed(42)
PositiveIndex = np.random.choice(np.arange(len(Positive)),46200)
PositiveData = []
for i in PositiveIndex:
    PositiveData.append(Positive[i])
PositiveTrain = np.array(PositiveData[:45000])      #44000个
PositiveTest = np.array(PositiveData[45000:])       #1100个
PositiveTrainLabel = []                             #Positive加标签
for i in range(len(PositiveTrain)):
    PositiveTrainLabel.append(1)
PositiveTestLabel = []
for i in range(len(PositiveTest)):
    PositiveTestLabel.append(1)

NegativeData = np.random.permutation(Negative)
NegativeTest = NegativeData[:1120]                  #1080个
NegativeTrain = NegativeData[1120:]
NegativeTestLabel = []
for i in range(len(NegativeTest)):
    NegativeTestLabel.append(-1)


#扩充Negative数量,
temp = NegativeTrain
for i in range(9):
    NegativeTrain = np.vstack((NegativeTrain,temp)) # 43960个
NegativeTrainLabel = []
for i in range(len(NegativeTrain)):
    NegativeTrainLabel.append(-1)


#将训练数据合并，并打乱
TrainData_Temp = list(PositiveTrain) + list(NegativeTrain)
TrainLabel_Temp = PositiveTrainLabel + NegativeTrainLabel
TrainIndex = np.arange(len(TrainLabel_Temp ))
TrainIndex = np.random.permutation(TrainIndex)
TrainData = []
TrainLabel = []
for i in TrainIndex:
    TrainData.append(TrainData_Temp[i])
    TrainLabel.append(TrainLabel_Temp[i])


#将测试数据合并，并打乱
TestData_Temp = list(PositiveTest) + list(NegativeTest)
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
              "C":[0.001, 0.01, 0.1, 1, 10, 100]
             }
print("parametes : {}".format(param_grid))
#clf = SVC(kernel="rbf", C = 1.0, gamma= "auto")
grid_search = GridSearchCV(SVC(),param_grid,cv=5)
grid_search.fit(TrainData, TrainLabel)
print("训练完成")


#评价
print("Test score : {}".format(grid_search.score(TestData, TestLabel)))
print("Best parameters : {}".format(grid_search.best_params_))
print("Best score on train : {}".format(grid_search.best_score_))