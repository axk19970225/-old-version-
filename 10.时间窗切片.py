import h5py
import numpy as np

TimeWidth = 32                    #时间窗长度
BreakTime = 30
InvalidTime = 5
standard = {                      #归一化基准
            'ip': 225,
            'Bt': 2.05,
            'axuv_ca_01': 1.3,
            'sxr_cb_024': 1.5,
            'sxr_cc_049': 4,
            'vs_c3_aa001': 5,
            'vs_ha_aa001': 1,
            'exsad1': 6, 'exsad2': 2.5, 'exsad4': 1.3, 'exsad7': 4, 'exsad8': 1, 'exsad10': 6,
            'Ivfp': 3,
            'Ihfp': 1,
            'MA_POL_CA01T': 2, 'MA_POL_CA02T': 2, 'MA_POL_CA03T': 2, 'MA_POL_CA05T': 2, 'MA_POL_CA06T': 2,
            'MA_POL_CA07T': 2, 'MA_POL_CA19T': 2, 'MA_POL_CA20T': 2, 'MA_POL_CA21T': 2, 'MA_POL_CA22T': 2,
            'MA_POL_CA23T': 2, 'MA_POL_CA24T': 2
            }

print(len(standard))

#训练信号
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
            r'\MA_POL_CA07T',r'\MA_POL_CA19T', r'\MA_POL_CA20T', r'\MA_POL_CA21T', r'\MA_POL_CA22T', r'\MA_POL_CA23T', r'\MA_POL_CA24T'
        ]
all_tags =  lf_tags + hf_tags
for i in range(len(all_tags)):       #去掉\
    all_tags[i] = all_tags[i][1:]


TrainBreak = np.load(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\训练集\BreakTrain.npy")
TrainNormal = np.load(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\训练集\NormalTrain.npy")



n = 1
PositiveAllData = np.mat(np.ones((1, len(all_tags)*TimeWidth)))
for num in TrainNormal:
    print("第{}个".format(n))
    n += 1
    g = h5py.File(r"H:\Train data\{}.hdf5".format(num), 'r')
    data = g.get("ip")
    step = len(data) - TimeWidth + 1
    incident = np.mat(np.ones((step, 1)))
    for tag in all_tags:
        data = g.get(tag)
        data = (np.array(data)) / (standard[tag])
        chip = []
        for i in range(step):
            chip.append(data[i : (i + TimeWidth)])
        chip = np.mat(chip)
        incident = np.hstack((incident,chip))
    incident = np.delete(incident, 0, axis=1)
    print(incident.shape)
    g.close()
    PositiveAllData = np.vstack((PositiveAllData,incident))
PositiveAllData = np.delete(PositiveAllData, 0, axis=0)


NegativeAllData = np.mat(np.ones((1, len(all_tags)*TimeWidth)))
for num in TrainBreak:
    print("第{}个".format(n))
    n += 1
    g = h5py.File(r"H:\Train data\{}.hdf5".format(num), 'r')
    data = g.get("ip")
    StepNormal = len(data) - TimeWidth - BreakTime + 1
    incident = np.mat(np.ones((StepNormal, 1)))
    for tag in all_tags:
        data = g.get(tag)
        data = (np.array(data)) / (standard[tag])
        chip = []
        for i in range(StepNormal):
            chip.append(data[i : (i + TimeWidth)])
        chip = np.mat(chip)
        incident = np.hstack((incident,chip))
    incident = np.delete(incident, 0, axis=1)
    print(incident.shape)
    PositiveAllData = np.vstack((PositiveAllData,incident))
    StepInvalid = len(data) - TimeWidth - InvalidTime + 1
    NegativeIncident = np.mat(np.ones(((BreakTime - InvalidTime), 1)))
    for tag in all_tags:
        data = g.get(tag)
        data = (np.array(data)) / (standard[tag])
        chip = []
        for i in range(StepNormal,StepInvalid):
            chip.append(data[i : (i + TimeWidth)])
        chip = np.mat(chip)
        NegativeIncident = np.hstack((NegativeIncident, chip))
    NegativeIncident = np.delete(NegativeIncident, 0, axis=1)
    NegativeAllData = np.vstack((NegativeAllData,NegativeIncident))
    g.close()


PositiveAllData = np.array(PositiveAllData)
NegativeAllData = np.array(NegativeAllData)
print(PositiveAllData.shape)
print(NegativeAllData.shape)

np.save(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\TrainPositive.npy",PositiveAllData)
np.save(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\TrainNegative.npy",NegativeAllData)