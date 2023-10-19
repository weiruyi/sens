import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import Spectrogram as sp

file_path = 'D:/hnu/python/IMU_dataset/mmm.csv'

df = pd.read_csv(file_path, header=None)

signal_x = df.iloc[:, 1].tolist()
signal_y = df.iloc[:, 2].tolist()
signal_z = df.iloc[:, 3].tolist()
length = len(signal_x)

fs = 417
t = np.linspace(0, length/fs, length, endpoint=False)  # 时间序列

# plt.plot(t, signal_z)
# plt.show()

# # ---------------------2、高通滤波器------------------

L = 1000
R = length - 1000

signal_x = signal_x[L:R]
signal_y = signal_y[L:R]
signal_z = signal_z[L:R]

sp.generateMap('1', signal_x, signal_y, signal_z)

"""

##
# 创建一个高通滤波器
def highpass_filter(signal, cutoff_freq, fs):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='high', analog=False)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

# 对信号进行高通滤波，消除人体运动的影响
fpass = 20
signal_x = highpass_filter(signal_x, fpass, fs)
signal_y = highpass_filter(signal_y, fpass, fs)
signal_z = highpass_filter(signal_z, fpass, fs)

length -= 100

# t = np.linspace(0, length/fs, length, endpoint=False)  # 时间序列
# plt.plot(t, signal_z)
# plt.show()

signal_x = signal_x[20:R]
signal_y = signal_y[20:R]
signal_z = signal_z[20:R]
length -= 20

# t = np.linspace(0, length/fs, length, endpoint=False)  # 时间序列
# plt.plot(t, signal_z)
# plt.show()

# 归一化
signal_x = (signal_x - np.mean(signal_x))/np.var(signal_x)
signal_y = (signal_x - np.mean(signal_y))/np.var(signal_y)
signal_z = (signal_x - np.mean(signal_z))/np.var(signal_z)


# 第一次粗粒度切割1（可认为是振动检测）
# 对信号进行切割
# 基本思想：找到最大峰值，然后向左、向右移动固定的距离（可以改为动态法处理，过零率）
# 找到最大峰值和其索引
Mmax = np.max(signal_z)
Mindex = np.argmax(signal_z)

# 设置左右移动的距离
L_cut = 5
R_cut = 8
# 计算切割范围的索引
index = [Mindex - L_cut, Mindex + R_cut]

# 创建切割后的子信号
seg_x = signal_z[index[0]:index[1]]
seg_z = seg_x.copy()

# 对 seg_z 进行平均移动（示例中的 Avermoving 函数未提供，你需要自行实现）
# 这里使用简单的均值滤波作为示例
def Avermoving(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

seg_z = Avermoving(seg_z, 3)

# # 绘制时域信号图
# plt.plot(seg_z, 'b')
# plt.title('time_domain')
# plt.show()




from scipy.fft import fft
ft = fft(seg_z)
mag = np.abs(ft)
fre = np.linspace(0, fs, len(mag))
plt.plot(fre, mag)
plt.title("magnitude spectrum")
plt.xlabel("Hz")
plt.ylabel("magnitude")
plt.show()

"""



