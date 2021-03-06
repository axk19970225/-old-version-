import numpy as np
import math
import h5py
from scipy import signal as signal
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.externals import joblib
import joblib




#对10Hz采样的信号进行加工处理
def process_10Hz(data):
    processed_data = []
    section = math.ceil(len(data) / 10)
    for j in range(section):
        if j < (section - 1):
            mean = np.mean(data[(10 * j) :(10 * j + 10)])
            processed_data.append(mean)
        else:
            mean = np.mean(data[(10 * j):])
            processed_data.append(mean)
    return processed_data


#对250Hz采样的信号进行加工处理
def process_250Hz(data):
    processed_data = []
    section = math.ceil(len(data) / 250)
    for j in range(section):
        if j < (section - 1):
            mean = np.mean(data[(250 * j):(250 * j + 250)])
            processed_data.append(mean)
        else:
            mean = np.mean(data[(250 * j):])
            processed_data.append(mean)
    return processed_data


#对500Hz采样的信号进行加工处理
def process_500Hz(data):
    processed_data = []
    section = math.ceil(len(data) / 500)
    for j in range(section):
        if j < (section - 1):
            mean = np.mean(data[(500 * j):(500 * j + 500)])
            processed_data.append(mean)
        else:
            mean = np.mean(data[(500 * j):])
            processed_data.append(mean)
    return processed_data


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


def reduce_sampling(shot):
    reduce_sampling_signal = {}
    g = h5py.File(r"/home/axk/chipsvm/data/{}.hdf5".format(shot), 'r')
    dataset = g.get("ip")
    data = list(dataset["data"])
    time = list(dataset["time"])
    DownTime, shot_flag = down_time(data)
    if shot_flag == -1:
        print("此炮为正常炮")
        shot_label = -1
        print("DownTime : {}".format(DownTime))
        for shottag in all_tags:
            dataset = g.get(shottag)
            data = np.array(dataset["data"])
            time = np.array(dataset["time"])
            data = data[time > 0.05]
            time = time[time > 0.05]
            data = data[time <= DownTime]
            time = time[time <= DownTime]
            datatemp = data[time < 0.06]
            if len(datatemp) < 200:
                processed_data = process_10Hz(data)
            elif len(datatemp) < 3000:
                processed_data = process_250Hz(data)
            else:
                processed_data = process_500Hz(data)
            reduce_sampling_signal.update({shottag:processed_data})
    else:
        print("此炮为破裂炮")
        shot_label = 1
        print("DownTime : {}".format(DownTime))
        for shottag in all_tags:
            dataset = g.get(shottag)
            data = np.array(dataset["data"])
            time = np.array(dataset["time"])
            data = data[time > 0.05]
            time = time[time > 0.05]
            data = data[time <= DownTime]
            time = time[time <= DownTime]
            datatemp = data[time < 0.06]
            if len(datatemp) < 200:
                processed_data = process_10Hz(data)
            elif len(datatemp) < 3000:
                processed_data = process_250Hz(data)
            else:
                processed_data = process_500Hz(data)
            reduce_sampling_signal.update({shottag: processed_data})
    g.close()
    return reduce_sampling_signal, shot_label


def time_chipping(shot):
    signal, shot_label = reduce_sampling(shot)
    data = signal["ip"]
    step = len(data) - TimeWidth + 1
    incident = np.mat(np.ones((step, 1)))
    for tag in all_tags:
        data = signal[tag]
        data = (np.array(data)) / (standard[tag])
        chip = []
        for i in range(step):
            chip.append(data[i: (i + TimeWidth)])
        chip = np.mat(chip)
        incident = np.hstack((incident, chip))
    incident = np.delete(incident, 0, axis=1)
    incident = np.array(incident)
    return incident




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


BreakNum = np.load(r"/home/axk/chipsvm/BreakTest.npy")
NormalNum = np.load(r"/home/axk/chipsvm/NormalTest.npy")


clf = joblib.load(r"/home/axk/chipsvm/SVMmodal/svmModel1c0.1.plk")
n = 1
for shot in NormalNum:
    chip = time_chipping(shot)
    print("第{}炮".format(n))
    n += 1
    print(chip.shape)
    ChipPredict = clf.predict(chip)
    print(ChipPredict)
    positive = ChipPredict[ChipPredict == 1].size
    negative = ChipPredict[ChipPredict == -1].size
    print(positive)
    print(negative)
    plt.figure(shot)
    plt.plot(ChipPredict)
    plt.savefig(r"/home/axk/shotsvm/result/1c0.1/single/normalpredictplot/{}".format(shot))
    plt.close()

