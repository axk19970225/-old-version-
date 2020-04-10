import numpy as np

NormalShot = np.load(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\TrainNormal.npy")
BreakShot = np.load(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\TrainBreak.npy")
NormalShot = list(NormalShot)
BreakShot = list(BreakShot)
print(len(NormalShot))
print(len(BreakShot))


np.random.seed(42)
BreakTest = np.random.choice(BreakShot, 50, replace = False)
NormalTest = np.random.choice(NormalShot, 50, replace = False)
BreakTest = list(BreakTest)
NormalTest = list(NormalTest)


for i in BreakTest:
    BreakShot.remove(i)
BreakTrain = BreakShot

for i in NormalTest:
    NormalShot.remove(i)
NormalTrain = NormalShot

print("NormalTrain : {}".format(len(NormalTrain)))
print("BreakTrain : {}".format(len(BreakTrain)))
print("NormalTest : {}".format(len(NormalTest)))
print("BreakTest : {}".format(len(BreakTest)))

np.save(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\测试集\NormalTest.npy",NormalTest)
np.save(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\测试集\BreakTest.npy",BreakTest)
np.save(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\训练集\NormalTrain.npy",NormalTrain)
np.save(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\训练集\BreakTrain.npy",BreakTrain)
