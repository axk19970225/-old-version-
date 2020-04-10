import h5py
import numpy as np
import matplotlib.pyplot as plt


#训练的信号名
lf_tags = [
            r'\ip',
            r'\Bt',
            r'\axuv_ca_01',
            r'\sxr_cb_024',
            r'\vs_c3_aa001',
            r'\vs_ha_aa001',
            r'\sxr_cc_049',
            r'\exsad1', r'\exsad2', r'\exsad4', r'\exsad7', r'\exsad8', r'\exsad10',
            r'\Ivfp', r'\Ihfp'
        ]

hf_tags = [
            r'\MA_POL_CA01T', r'\MA_POL_CA02T', r'\MA_POL_CA03T', r'\MA_POL_CA05T', r'\MA_POL_CA06T',
            r'\MA_POL_CA07T',r'\MA_POL_CA19T', r'\MA_POL_CA20T', r'\MA_POL_CA21T', r'\MA_POL_CA22T',
            r'\MA_POL_CA23T', r'\MA_POL_CA24T'
        ]
all_tags =  lf_tags + hf_tags
print(len(all_tags))

for i in range(len(all_tags)):
    all_tags[i] = all_tags[i][1:len(all_tags[i])]
print(all_tags)


TrainNumber = np.load(r"C:\Users\艾鑫坤\Desktop\破裂炮数据\新一轮\ValidNumber.npy")
TrainNumber = list(TrainNumber)

print(len(TrainNumber))
print(TrainNumber)

n = 1
IncompleteNumber = []
for i in TrainNumber:
    print("第{}个".format(n))
    n += 1
    g = h5py.File(r"I:\predict data\{}.hdf5".format(i),'r')
    ShotTag = list(g.keys())
    for j in all_tags:
        if j in ShotTag:
            continue
        else:
            IncompleteNumber.append(i)
            break
    g.close()

print(len(IncompleteNumber))
for i in IncompleteNumber:
    for j in TrainNumber:
        if int(i) == int(j):
            TrainNumber.remove(j)
print(len(TrainNumber))
print(TrainNumber)

np.save(r"C:\Users\艾鑫坤\Desktop\破裂炮数据\新一轮\信号一样.npy",TrainNumber)
