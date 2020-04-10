from scipy import signal as signal
import h5py
import numpy as np
import matplotlib.pyplot as plt


TrainNormal = np.load(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\TrainNormal.npy")
TrainBreak = np.load(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\train\TrainBreak.npy")
TrainNormal = list(TrainNormal)
TrainBreak = list(TrainBreak)
print(len(TrainNormal))
print(len(TrainBreak))
TrainNumber = TrainNormal + TrainBreak

#TrainNumber = np.load(r"C:\Users\艾鑫坤\Desktop\破裂炮数据\新一轮\信号一样.npy")

normalnum = []
breaknum = []
n = 1
mistake = []
for num in TrainNumber:
    print("第{}个".format(n))
    n += 1
    g = h5py.File(r"H:\predict data\{}.hdf5".format(num), 'r')
    dataset = g.get("ip")
    data = list(dataset["data"])
    time = list(dataset["time"])
    g.close()
    # 低通滤波
    ip = data
    ba = signal.butter(8, 0.04, 'lowpass')
    ip_ = signal.filtfilt(ba[0], ba[1], ip)
    # 中值滤波
    ip = signal.medfilt(ip, 15)
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
    k = []  # 斜率
    for i in range(190):
        k_i = (data[50 * i + 49] - data[50 * i]) / 0.005    #每隔50ms取一次斜率
        k.append(k_i)
    if shot_label == 1:         #判断炮是正常炮还是破裂炮，-4000是将取的斜率画图观察得到的一个阈值
        print("此炮为正常炮")
        normalnum.append(num)
        DownTime = (start + 500)/10000
        DownData = 150                              #下降时刻的幅值不重要，随便写的，主要还是下降时间重要
        plt.figure(num)
        print(len(data))
        print(len(time))
        plt.plot(time, data, 'g', DownTime, DownData, 'ro')
        plt.savefig(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\正常炮\{}".format(num))
        plt.close()
    else:
        try:
            #我写的判断破裂炮的下降时间，精确到1ms
            print("此炮为破裂炮")
            for i in range(30, 190):
                if (k[i] > -4000) and (k[i + 1] < -4000):
                    for j in range(50):
                        error = data[50 * i + 50 + j + 1] - data[50 * i + 50 + j]
                        if error < -5:
                            down_time = ((50 * i + 50 + j) / 10000) + 0.05
                            down_data = data[50 * i + 50 + j]
                            break
                    break
            # 取到小数点后3位
            down_data = round(down_data, 3)
            plt.figure(num)
            plt.plot(time, data, 'g', down_time, down_data, 'ro')
            plt.savefig(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\破裂炮\{}".format(num))
            plt.close()
            breaknum.append(num)
        except Exception as err:
            print("错误炮 ： {}".format(num))
            mistake.append(num)
            print(mistake)
            continue


print(len(normalnum))
print(len(breaknum))
# np.save(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\Normal.npy",normalnum)
# np.save(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\Break.npy",breaknum)
# np.save(r"C:\Users\ASJTL\Desktop\破裂炮数据\新一轮\error.npy",mistake)