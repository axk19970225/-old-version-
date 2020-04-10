from scipy import signal as signal
import h5py
import numpy as np
import matplotlib.pyplot as plt


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
    return  DownTime


# TrainNormal = np.load(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\Normal.npy")
# TrainBreak = np.load(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\Break.npy")
# TrainNormal = list(TrainNormal)
# TrainBreak = list(TrainBreak)
# print(len(TrainNormal))
# print(len(TrainBreak))
# TrainNumber = TrainNormal + TrainBreak

TrainNumber = np.load(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\Break.npy")
print(len(TrainNumber))

breakvp = []

n = 1
for num in TrainNumber:
    print("第{}个".format(n))
    n += 1
    g = h5py.File(r"H:\predict data\{}.hdf5".format(num), 'r')
    dataset = g.get("vp2")
    data = list(dataset["data"])
    time = list(dataset["time"])
    ip = g.get("ip")
    ipdata = list(ip["data"])
    g.close()
    # 低通滤波
    ba = signal.butter(8, 0.01, "lowpass")
    fdata = signal.filtfilt(ba[0], ba[1], data)
    if max(fdata) < 0.015:
        breakvp.append(num)

print(breakvp)
print(len(breakvp))
np.save(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\breakvp.npy",breakvp)



