"""
读取音频文件信息
"""
import librosa
import librosa.display
import numpy as np
import soundfile as f
from scipy.io import wavfile
import wave
from matplotlib import pyplot as plt


def plot_timeDomain(wave_form, sr):
    """
    绘制时域图
    :param wave_form: 音频信号
    :param sr: 采样率
    :return: None
    """
    # 计算时间轴
    duration = len(wave_form) / sr
    time = np.linspace(0, duration, len(wave_form))

    # 绘制时域图
    plt.figure(figsize=(10, 4))
    plt.plot(time, wave_form)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


def plot_fftSpec(wave_form, sr):
    """
    绘制傅里叶变换后的频谱图
    :param wave_form: 音频信号
    :param sr: 采样率
    :return: None
    """
    # 计算短时傅里叶变换
    D = librosa.stft(wave_form)
    # 将幅度谱转换为分贝刻度
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # 绘制频谱图
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D_db, sr=sr, hop_length=5, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('STFT Spectrogram')
    plt.tight_layout()
    plt.show()

def plot_mfcc(waveform,  sample_rate):
    # 计算MFCC特征
    mfcc = librosa.feature.mfcc(waveform, sr=sample_rate)
    print(mfcc.shape)
    # 绘制MFCC频谱图
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    plt.show()

# # 方法一：soundfile
# clean_sig, sr = f.read('./data/airleft.wav')    # clean_sig：内容，sr：采样率
#
# print('内容：' + str(clean_sig))
# print('形状：' + str(clean_sig.shape))
# print('采样率：' + str(sr))
# print('类型：' + str(clean_sig.dtype))


# # 方法二：scipy.io
# sr_2, clean_sig_2 = wavfile.read('./data/left.wav')
# # 采样率为第一个参数，和soundfile读取不同
# print('内容：' + str(clean_sig_2))
# print('形状：' + str(clean_sig_2.shape))
# print('采样率：' + str(sr_2))
# print('类型：' + str(clean_sig_2.dtype))
# # plot_timeDomain(clean_sig_2, sr_2)




# 方法三
clean_sig , sr = librosa.load('../dataset/prior/wry/帮助我.wav')
print('内容：' + str(clean_sig))
print('形状：' + str(clean_sig.shape))
print('采样率：' + str(sr))
print('类型：' + str(clean_sig.dtype))
# plot_mfcc(clean_sig, sr)
# plot_timeDomain(clean_sig,sr)
plot_fftSpec(clean_sig,sr)
# stft_s = librosa.stft(clean_sig)
# stft_s = stft_s[20:501,:]
# print(stft_s.shape)
# plt.figure(figsize=(10, 6))
# librosa.display.specshow(stft_s, hop_length=5, x_axis='time', y_axis='linear')
# plt.colorbar(format='%+2.0f dB')
# plt.title('STFT Spectrogram')
# plt.tight_layout()
# plt.show()



