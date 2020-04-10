import numpy as np
import math
import h5py
import openpyxl
from scipy import signal as signal
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.externals import joblib
import joblib

NegativeSect = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
PositiveSect = [0,1,2,3,4,5]

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



#获取测试炮和其对应的标签，正常炮200个，破裂炮200个
np.random.seed(4)
TestBreak = np.load(r"/home/axk/chipsvm/BreakTest.npy")
TestNormal = np.load(r"/home/axk/chipsvm/NormalTest.npy")
# TestBreak = np.random.choice(TestBreak, 200)
# TestNormal = np.random.choice(TestNormal, 200)
BreakLabel = []
for i in range(len(TestBreak)):
    BreakLabel.append(1)
BreakLabel = np.array(BreakLabel)

NormalLabel = []
for i in range(len(TestNormal)):
    NormalLabel.append(-1)
NormalLabel = np.array(NormalLabel)
TestTemp = list(TestBreak) + list(TestNormal)
LabelTemp = list(BreakLabel) + list(NormalLabel)
TrainIndex = np.arange(len(LabelTemp))
TrainIndex = np.random.permutation(TrainIndex)
test = []
label = []
for i in TrainIndex:
    test.append(TestTemp[i])
    label.append(LabelTemp[i])
label =np.array(label)
print("label : ")
print(len(label))

#对测试炮进行预测
clf = joblib.load(r"/home/axk/chipsvm/SVMmodal/svmModel1c0.1.plk")
accuarymatrix = []
precisionmatrix = []
recallmatrix = []
FPRmatrix = []
Advancelist = h5py.File(r"/home/axk/shotsvm/result/1c0.1/matrix/advancehdf5/Advancematrix1vs0.1.hdf5")
for PositiveThreshold in PositiveSect:
    Negativeaccuary = []
    Negativeprecision = []
    Negativerecall = []
    NegativeFPR = []
    for NegativeThreshold in NegativeSect:
        Threshold = NegativeThreshold + PositiveThreshold
        Advancename = str(PositiveThreshold) + "vs" + str(NegativeThreshold)
        print(Advancename)
        n = 1
        shotlabel = []
        Advance = []
        for shot in test:
            flag = -1
            chip = time_chipping(shot)
            chiplength = len(chip)
            print("第{}炮".format(n))
            n += 1
            ChipPredict = []
            for i in range(len(chip)):
                count_positive = 0
                count_negative = 1
                ChipPred_Label = clf.predict([chip[i]])
                ChipPredict.append(ChipPred_Label)
                if len(ChipPredict) > Threshold:
                    Sect = np.array(ChipPredict[-Threshold:])
                    num = Sect[Sect == 1].size
                    if (num / len(Sect)) > 0.5:
                        if Sect[0] == 1:
                            for j in range(1, len(Sect)):
                                if ((Sect[j - 1] == 1) and (Sect[j] == 1)):
                                    count_negative += 1
                                    if count_negative > NegativeThreshold:
                                        flag = 1
                                        break
                                elif (((Sect[j - 1] == 1) and (Sect[j] == -1)) or (
                                    (Sect[j - 1] == -1) and (Sect[j] == -1))):
                                    count_positive += 1
                                    if count_positive >= PositiveThreshold:
                                        flag = -1
                                        count_negative = 0
                                        break
                                elif ((Sect[j - 1] == -1) and (Sect[j] == 1)):
                                    count_positive = 0
                                    count_negative += 1
                                    if count_negative > NegativeThreshold:
                                        flag = 1
                                        break
                if flag == 1:
                    shotlabel.append(flag)
                    AdvanceTime = chiplength - i - 1
                    print("提前时间 ： {}".format(AdvanceTime))
                    Advance.append(AdvanceTime)
                    break
            if flag == -1:
                shotlabel.append(flag)
            print(shotlabel[-1])
        print(shotlabel)
        print(label)
        Advancelist.create_dataset("{}".format(Advancename),data = Advance)
        print("Advance长度 ：{}".format(len(Advance)))
        plt.figure(Advancename)
        plt.plot(Advance, 'g')
        plt.savefig(r"/home/axk/shotsvm/result/1c0.1/matrix/advanceplot/{}".format(Advancename))
        plt.close()
        # 评价
        shotlabel = np.array(shotlabel)
        correct = 0
        for i in range(len(label)):
            if int(label[i]) == int(shotlabel[i]):
                correct += 1
        accuary = correct / len(shotlabel)
        Negativeaccuary.append(accuary)
        print(accuary)
        #混淆矩阵
        conf = confusion_matrix(label, shotlabel)
        #precision
        precision = precision_score(label, shotlabel)
        Negativeprecision.append(precision)
        #recall
        recall = recall_score(label, shotlabel)
        Negativerecall.append(recall)
        #FPR
        FPR = conf[0, 1] / (conf[0, 1] + conf[0, 0])
        NegativeFPR.append(FPR)
    accuarymatrix.append(Negativeaccuary)
    precisionmatrix.append(Negativeprecision)
    recallmatrix.append(Negativerecall)
    FPRmatrix.append(NegativeFPR)
Advancelist.close()


workbook = xlsxwriter.Workbook(r"/home/axk/shotsvm/result/1c0.1/matrix/result.xlsx")
print("accuary : ")
print(accuarymatrix)
worksheet1 = workbook.add_worksheet('accuary')
for i in range(len(NegativeSect)):
    worksheet1.write(0, (i + 1), NegativeSect[i])
for i in range(len(PositiveSect)):
    worksheet1.write((i + 1), 0, PositiveSect[i])
for i in range(len(PositiveSect)):
    for j in range(len(NegativeSect)):
        worksheet1.write((i + 1), (j + 1), accuarymatrix[i,j])
np.save(r"/home/axk/shotsvm/result/1c0.1/matrix/accuarymatrix.npy",accuarymatrix)

print("precision : ")
print(precisionmatrix)
worksheet2 = workbook.add_worksheet('precision')
for i in range(len(NegativeSect)):
    worksheet2.write(0, (i + 1), NegativeSect[i])
for i in range(len(PositiveSect)):
    worksheet2.write((i + 1), 0, PositiveSect[i])
for i in range(len(PositiveSect)):
    for j in range(len(NegativeSect)):
        worksheet2.write((i + 1), (j + 1), precisionmatrix[i,j])
np.save(r"/home/axk/shotsvm/result/1c0.1/matrix/precisionmatrix.npy",precisionmatrix)

print("recall : ")
print(recallmatrix)
worksheet3 = workbook.add_worksheet('recall')
for i in range(len(NegativeSect)):
    worksheet3.write(0, (i + 1), NegativeSect[i])
for i in range(len(PositiveSect)):
    worksheet3.write((i + 1), 0, PositiveSect[i])
for i in range(len(PositiveSect)):
    for j in range(len(NegativeSect)):
        worksheet3.write((i + 1), (j + 1), recallmatrix[i,j])
np.save(r"/home/axk/shotsvm/result/1c0.1/matrix/recallmatrix.npy",recallmatrix)

print("FPR")
print(FPRmatrix)
worksheet4 = workbook.add_worksheet('FPR')
for i in range(len(NegativeSect)):
    worksheet4.write(0, (i + 1), NegativeSect[i])
for i in range(len(PositiveSect)):
    worksheet4.write((i + 1), 0, PositiveSect[i])
for i in range(len(PositiveSect)):
    for j in range(len(NegativeSect)):
        worksheet4.write((i + 1), (j + 1), FPRmatrix[i,j])
np.save(r"/home/axk/shotsvm/result/1c0.1/matrix/FPRmatrix.npy",FPRmatrix)
workbook.close()

