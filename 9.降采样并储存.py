import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as signal


#下降时间
def down_time(data):
    ip = data
    ba = signal.butter(8, 0.04, 'lowpass')
    ip_ = signal.filtfilt(ba[0], ba[1], ip)
    start = 0
    end = 0
    for i in range(len(ip_ - 640)):
        if ip_[i] > ip_[i + 20] > ip_[i + 40] and ip_[i] * 0.95 > ip_[i + 160] and ip_[i] * 0.9 > ip_[i + 320] \
                and ip_[i] * 0.8 > ip_[i + 640]:
            start = i
            break
    for i in range(start, len(ip_ - 640)):
        if ip_[i] <= 10:
            end = i
            break
    if start and end:
        if end - start < 600:
            shot_label = -1
        else:
            shot_label = 1
    k = []
    for i in range(190):
        k_i = (data[50 * i + 49] - data[50 * i]) / 0.005
        k.append(k_i)
    if shot_label == 1:
        # print("此炮为正常炮")
        DownTime = (start + 500) / 10000
        DownData = data[start]
    else:
        # print("此炮为破裂炮")
        for i in range(30, 190):
            if (k[i] > -4000) and (k[i + 1] < -4000):
                for j in range(50):
                    error = data[50 * i + 50 + j + 1] - data[50 * i + 50 + j]
                    if error < -5:
                        DownTime = ((50 * i + 50 + j) / 10000) + 0.05
                        down_data = data[50 * i + 50 + j]
                        break
                break
    return  DownTime, shot_label


#对10Hz采样的信号进行加工处理
def process_10Hz(data):
    processed_data = []
    section = math.ceil(len(data) / 10)
    for j in range(section):
        if j < (section - 1):
            # sum = 0
            # for k in range(10):
            #     sum = sum + data[10 * j + k]
            # mean = sum / 10
            mean = np.mean(data[(10 * j) :(10 * j + 10)])
            processed_data.append(mean)
        else:
            # sum = 0
            # for k in range(len(data[(10 * j):])):
            #     sum = sum + data[10 * j + k]
            # mean = sum / (len(data[(10 * j):]))
            mean = np.mean(data[(10 * j):])
            processed_data.append(mean)
    return processed_data


#对250Hz采样的信号进行加工处理
def process_250Hz(data):
    processed_data = []
    section = math.ceil(len(data) / 250)
    for j in range(section):
        if j < (section - 1):
            # sum = 0
            # for k in range(250):
            #     sum = sum + data[250 * j + k]
            # mean = sum / 250
            mean = np.mean(data[(250 * j):(250 * j + 250)])
            processed_data.append(mean)
        else:
            # sum = 0
            # for k in range(len(data[(250 * j):])):
            #     sum = sum + data[250 * j + k]
            # mean = sum / (len(data[(250 * j):]))
            mean = np.mean(data[(250 * j):])
            processed_data.append(mean)
    return processed_data


#对500Hz采样的信号进行加工处理
def process_500Hz(data):
    processed_data = []
    section = math.ceil(len(data) / 500)
    for j in range(section):
        if j < (section - 1):
            # sum = 0
            # for k in range(500):
            #     sum = sum + data[500 * j + k]
            # mean = sum / 500
            mean = np.mean(data[(500 * j):(500 * j + 500)])
            processed_data.append(mean)
        else:
            # sum = 0
            # for k in range(len(data[(500 * j):])):
            #     sum = sum + data[500 * j + k]
            # mean = sum / (len(data[(500 * j):]))
            mean = np.mean(data[(500 * j):])
            processed_data.append(mean)
    return processed_data


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


#调取可训练的炮号
BreakTrain = np.load(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\训练集\BreakTrain.npy")
NormalTrain = np.load(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\训练集\NormalTrain.npy")
BreakTest = np.load(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\测试集\BreakTest.npy")
NormalTest = np.load(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\测试集\NormalTest.npy")
BreakTrain = list(BreakTrain)
NormalTrain = list(NormalTrain)
BreakTest = list(BreakTest)
NormalTest = list(NormalTest)
# TrainShot = BreakTrain + NormalTrain + BreakTest + NormalTest
TrainShot = BreakTrain + NormalTrain
print(TrainShot)


n = 1
for shot in TrainShot:
    file = h5py.File(r"H:\Train data\{}.hdf5".format(shot))
    print("第{}个".format(n))
    n += 1
    g = h5py.File(r"H:\predict data\{}.hdf5".format(shot), 'r')
    dataset = g.get("ip")
    data = list(dataset["data"])
    time = list(dataset["time"])
    DownTime, label = down_time(data)
    if label == 1:
        print("此炮为正常炮")
        for shottag in all_tags:
            dataset = g.get(shottag)
            data = np.array(dataset["data"])
            time = np.array(dataset["time"])
            data = data[time <= DownTime]
            time = time[time <= DownTime]
            data = data[time > 0.2]
            time = time[time > 0.2]
            datatemp = data[time < 0.21]
            if len(datatemp) < 200:
                processed_data = process_10Hz(data)
            elif len(datatemp) < 3000:
                processed_data = process_250Hz(data)
            else:
                processed_data = process_500Hz(data)
            file.create_dataset("{}".format(shottag),data = processed_data)
    else:
        print("此炮为破裂炮")
        for shottag in all_tags:
            dataset = g.get(shottag)
            data = np.array(dataset["data"])
            time = np.array(dataset["time"])
            data = data[time <= DownTime]
            time = time[time <= DownTime]
            data = data[time > 0.2]
            time = time[time > 0.2]
            datatemp = data[time < 0.21]
            if len(datatemp) < 200:
                processed_data = process_10Hz(data)
            elif len(datatemp) < 3000:
                processed_data = process_250Hz(data)
            else:
                processed_data = process_500Hz(data)
            file.create_dataset(shottag,data = processed_data)
    file.close()
    g.close()






