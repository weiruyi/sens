# from scipy.io import wavfile
# import noisereduce as nr
# import pyaudio
# import time
# import wave
# rate, data = wavfile.read("001.wav")
# _,noisy_part =  wavfile.read("noise.wav")
# SAMPLING_FREQUENCY=16000
# reduced_noise = nr.reduce_noise(y=data, y_noise=noisy_part, sr=SAMPLING_FREQUENCY)
#
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# RECORD_SECONDS = time
# WAVE_OUTPUT_FILENAME = "out_file.wav"
#
# with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(2)
#     wf.setframerate(RATE)
#     wf.writeframes(b''.join(reduced_noise))

from sklearn.decomposition import NMF
import scipy.io.wavfile as wav
import librosa
import numpy as np
from scipy import signal
import soundfile as sf
import wave
import logmmse
from sklearn.decomposition import PCA

def run2():
    # 假设您有一个名为 noisy_signal 的包含噪声的音频信号
    # noisy_signal, sr = librosa.load('D:/hnu/python/Gan/left_test.wav')
    noisy_signal, sr = librosa.load('denoised_signal.wav')

    # 设计低通滤波器
    cutoff_freq = 1500  # 截止频率，以Hz为单位
    cutoff_freq_low = 80
    sampling_freq = sr  # 采样频率，以Hz为单位
    nyquist_freq = 0.5 * sampling_freq  # 奈奎斯特频率++++++++++

    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(4, normalized_cutoff, btype='low')

    # 对信号应用低通滤波器
    denoised_signal_1 = signal.lfilter(b, a, noisy_signal)

    normalized_cutoff_low = cutoff_freq_low / nyquist_freq
    b, a = signal.butter(4, normalized_cutoff_low, btype='high')
    # 对信号应用高通滤波器
    denoised_signal = signal.lfilter(b, a, denoised_signal_1)

    # 打印降噪后的信号
    # print(denoised_signal)
    sf.write('denoised_signal2.wav', denoised_signal, sr)


# LogMMSE 降噪
def run3():
    path = 'D:/hnu/python/Gan/data_mod/gg_stf2.wav'
    f = wave.open(path, "r")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print("nchannels:", nchannels, "sampwidth:", sampwidth, "framerate:", framerate, "nframes:", nframes)
    data = f.readframes(nframes)
    f.close()
    data = np.fromstring(data, dtype=np.short)

    # 降噪
    data = logmmse.logmmse(data=data, sampling_rate=framerate)

    # 保存音频
    file_save = 'denoised_signal.wav'
    nframes = len(data)
    f = wave.open(file_save, 'w')
    f.setparams((1, 2, framerate, nframes, 'NONE', 'NONE'))  # 声道，字节数，采样频率，*，*
    # print(data)
    f.writeframes(data)  # outData
    f.close()


# Wiener 滤波器
def winner():
    # 读取带噪声的音频文件
    sample_rate, noisy_audio = wav.read('D:/hnu/python/Gan/data_mod/gg_stf2.wav')
    # 将音频信号转换为频域
    noisy_spec = np.fft.fft(noisy_audio)
    # 估计噪声功率谱
    noise_spec = np.abs(noisy_spec) ** 2
    # 估计信号功率谱
    signal_spec = noise_spec * 0.5  # 假设信号功率为噪声功率的一半
    # 计算谱减系数
    alpha = signal_spec / (signal_spec + noise_spec)
    # 对频域信号进行增益调整
    enhanced_spec = alpha * noisy_spec
    # 将增强后的频谱转换回时域
    enhanced_audio = np.fft.ifft(enhanced_spec).real.astype(np.int16)
    # 将增强后的音频保存为文件
    wav.write('denoised_signal.wav', sample_rate, enhanced_audio)


def pca():
    # 读取带噪声的音频文件
    sample_rate, noisy_audio = wav.read('D:/hnu/python/Gan/data_mod/gg_stf2.wav')
    from numpy.linalg import svd

    # 将音频信号转换为频域矩阵
    noisy_spec = np.abs(np.fft.fft(noisy_audio))

    # 执行SVD分解
    U, S, V = svd(noisy_spec)

    # 设置保留的奇异值数量（用于降噪）
    n_components = 100

    # 降噪信号的频谱
    denoised_spec = U[:, :n_components] @ np.diag(S[:n_components]) @ V[:n_components, :]

    # 将降噪的频谱转换回时域
    denoised_audio = np.fft.ifft(denoised_spec.T).real.astype(np.int16)

    # 将降噪后的音频保存为文件
    wav.write('denoised_audio.wav', sample_rate, denoised_audio)


if __name__ == "__main__":
    run3()
    # winner()
    # pca()