import h5py
import pprint
import numpy as np
import matplotlib.pyplot as plt

def similar(x,y):
    SameTag = []
    for i in x:
        for j in y:
            if i == j:
                SameTag.append(i)
    return SameTag

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
            r'\MA_POL_CA01T', r'\MA_POL_CA02T', r'\MA_POL_CA03T', r'\MA_POL_CA04T', r'\MA_POL_CA05T', r'\MA_POL_CA06T',
            r'\MA_POL_CA07T', r'\MA_POL_CA08T', r'\MA_POL_CA09T', r'\MA_POL_CA10T', r'\MA_POL_CA11T', r'\MA_POL_CA12T',
            r'\MA_POL_CA13T', r'\MA_POL_CA14T', r'\MA_POL_CA15T', r'\MA_POL_CA16T', r'\MA_POL_CA17T', r'\MA_POL_CA18T',
            r'\MA_POL_CA19T', r'\MA_POL_CA20T', r'\MA_POL_CA21T', r'\MA_POL_CA22T', r'\MA_POL_CA23T', r'\MA_POL_CA24T'
        ]
all_tags =  lf_tags + hf_tags

for i in range(len(all_tags)):
    all_tags[i] = all_tags[i][1:len(all_tags[i])]


# TrainNumber = np.load(r"C:\Users\艾鑫坤\Desktop\data\ValidNumber.npy")
TrainNumber = [1066444]
TrainNumber = list(TrainNumber)

print(len(TrainNumber))
print(TrainNumber)

result = {}
for i in all_tags:
    result.update({i:0})

n = 1
for i in TrainNumber:
    print("第{}个".format(n))
    n += 1
    g = h5py.File(r"I:\predict data\{}.hdf5".format(i),'r')
    ShotTag = list(g.keys())
    for j in all_tags:
        for tag in ShotTag:
            if j == tag :
                result[j] = result[j]+1
    g.close()

pprint.pprint(result)

"""
可用诊断：

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
            r'\MA_POL_CA07T'
            r'\MA_POL_CA19T', r'\MA_POL_CA20T', r'\MA_POL_CA21T', r'\MA_POL_CA22T', r'\MA_POL_CA23T', r'\MA_POL_CA24T'
        ]
all_tags =  lf_tags + hf_tags
"""



