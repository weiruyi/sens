import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import librosa
import soundfile


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


audio_path = '../dataset/prior/wry/保持深度.wav'
waveform, sr = librosa.load(audio_path)
# plt.subplot(211)
# plt.plot(range(len(waveform)), waveform)

filter_wave = butter_highpass_filter(waveform, 100, sr)
# plt.subplot(212)
# plt.plot(range(len(filter_wave)), filter_wave)
# plt.show()

soundfile.write('../dataset/denoised_wry_levelOff.wav', filter_wave, sr)
