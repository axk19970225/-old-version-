import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

filename = np.arange(1064034,1066648)
print(len(filename))

ValidNumber = []
n = 1
for i in filename:
    print("第{}个".format(n))
    n += 1
    try:
        g = h5py.File(r"I:\predict data\{}.hdf5".format(i),'r')
        dataset = g.get("ip")
        data = np.array(dataset["data"])
        time = np.array(dataset["time"])
        g.close()
        data_temp = data[time>0.2]
        ratio = len(data_temp[data_temp > 100]) / len(data_temp)
        if (max(data_temp) > 100)and(ratio>0.08):
            ValidNumber.append(i)
            print(len(ValidNumber))
    except Exception as err:
        continue

print(ValidNumber)
print(len(ValidNumber))
np.save(r"C:\Users\艾鑫坤\Desktop\破裂炮数据\新一轮\ValidNumber.npy",ValidNumber)