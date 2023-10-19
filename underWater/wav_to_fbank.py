"""
将wav转换成fbank特征
"""
import os
import re
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import librosa.display
import librosa
import numpy as np
from PIL import Image
from scipy.fft import fft


def getFbank(wavPath, sr, n_mels, hop_length, n_fft):
    # Load audio file
    waveform, sr = librosa.load(wavPath, sr=sr)
    # Compute FBank features with 40 filters and a window size of 25ms
    fbank = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft, win_length=512, window="blackman")
    # Convert power to dB scale
    fbank = librosa.power_to_db(fbank)
    return fbank


def saveFbank(fbank, outPath):
    np.save(outPath, fbank)

def spectrum_to_rgb(spectrum, n_mels):
    # Normalize the spectrum
    normalized_spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))

    # Scale the spectrum to 0-255 range
    scaled_spectrum = normalized_spectrum * 255

    # Create an RGB image
    rgb_image = np.zeros((n_mels, spectrum.shape[1], 3), dtype=np.uint8)

    # Map the spectrum channels to RGB channels
    rgb_image[:, :, 0] = scaled_spectrum  # Red channel
    rgb_image[:, :, 1] = scaled_spectrum  # Green channel
    rgb_image[:, :, 2] = scaled_spectrum  # Blue channel

    return rgb_image



def all():
    allWavs = os.listdir(wav_file)
    for e_wav in allWavs:
        e_wav_file = os.path.join(wav_file, e_wav)
        # e_label = e_wav.split('.')[0]
        e_label = re.split('\d', e_wav)[0]
        e_label_path = os.path.join(fbank_file, e_label)
        if not os.path.exists(e_label_path):
            os.mkdir(e_label_path)
        out_file = os.path.join(e_label_path, e_wav.split('.')[0] + '.png')
        e_fbank = getFbank(e_wav_file, sr, n_mels, hop_length, n_fft)
        fbank_img = spectrum_to_rgb(e_fbank, n_mels)
        # 将数组转换为图像对象
        image = Image.fromarray(fbank_img)
        # 保存为PNG格式
        image.save(out_file, format="PNG")


def sigle_wav():
     wav_path = os.path.join(wav_file, 'test_真棒.wav')
     fbank_f = getFbank(wav_path, sr, n_mels, hop_length, n_fft)
     fbank_img = spectrum_to_rgb(fbank_f, n_mels)
     # 将数组转换为图像对象
     image = Image.fromarray(fbank_img)
     # 保存为PNG格式
     image.save(r'D:\hnu\python\DenseNet\data_mod\test_真棒.png', format="PNG")

sr = 22050
n_mels = 26
hop_length = 5
n_fft = 512

wav_file = './data/'
fbank_file = './train_mod/'
# sigle_wav()

def compare():
    wavePath1 = '../dataset/data/wry_ascend_02.wav'
    wavePath2 = "../dataset/data/cyw_levelOff_03.wav"

    fbank_1 = getFbank(wavePath1, sr, n_mels, hop_length, n_fft)
    fbank_2 = getFbank(wavePath2, sr, n_mels, hop_length, n_fft)

    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(fbank_1.T, fbank_2.T)
    plt.imshow(similarity, cmap='hot', origin='lower')
    plt.colorbar()
    plt.xlabel('Audio 2 Frame')
    plt.ylabel('Audio 1 Frame')
    plt.title('Similarity Matrix')
    plt.show()


file_path = '../dataset/prior/wry/保持深度.wav'
# fbank = getFbank(file_path, sr, n_mels, hop_length, n_fft)
waveform, sr = librosa.load(file_path, sr=sr)
ft = fft(waveform)
mag = np.abs(ft)
fre = np.linspace(0, sr, len(mag))
plt.plot(fre, mag)
plt.title("magnitude spectrum")
plt.xlabel("Hz")
plt.ylabel("magnitude")
plt.show()