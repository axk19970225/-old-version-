import numpy as np

normalnum = np.load(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\TrainNormal.npy")
# breaknum = np.load(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\Break.npy")
normalnum = list(normalnum)
# breaknum = list(breaknum)
print(len(normalnum))
# print(len(breaknum))

PassNormal = [1064801, 1066109, 1065064]
# PassBreak = [1064867,1064870,1064921,1065401,1065541,1065549,1065553,1065556,1065559,1065563,1065586,1065589,1065593,1065605,1065612,1065622,1065633,
#              1065755,1065830,1065901,1065903,1065906,1065911,1065912,1065961,1065962,1065963,1065975,1065977,1065982,1065983,1065984,1065985,1065986,
#              1065987,1065988,1065991,1065995,1065996,1065999,1066462,1066477,1066478,1066495]

print(len(PassNormal))
# print(len(PassBreak))


for i in PassNormal:
    for j in normalnum:
        if int(i) == int(j):
            normalnum.remove(j)

# for i in PassBreak:
#     for j in breaknum:
#         if int(i) == int(j):
#             breaknum.remove(j)

print(len(normalnum))
# print(len(breaknum))

np.save(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\TrainNormal.npy",normalnum)
# np.save(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\TrainBreak.npy",breaknum)