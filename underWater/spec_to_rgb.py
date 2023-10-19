import os

import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from PIL import Image


def spectrum_to_rgb(spectrum):
    # Normalize the spectrum
    normalized_spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))

    # Scale the spectrum to 0-255 range
    scaled_spectrum = normalized_spectrum * 255

    # Create an RGB image
    rgb_image = np.zeros((26, spectrum.shape[1], 3), dtype=np.uint8)

    # Map the spectrum channels to RGB channels
    rgb_image[:, :, 0] = scaled_spectrum  # Red channel
    rgb_image[:, :, 1] = scaled_spectrum  # Green channel
    rgb_image[:, :, 2] = scaled_spectrum  # Blue channel

    return rgb_image


def create_dataset(data_files):
    dataset = os.listdir(data_files)
    return dataset


# if __name__ =='__main__':
#     dataPath = './data/'
#     outPuts = './train_stft/'
#     labels = create_dataset(dataPath)
#     for label in labels:
#         data_files = dataPath + label + '/'
#         out_files = outPuts + label + '/'
#         datasets = create_dataset(data_files)
#         for data in datasets:
#             data_file = data_files + data
#             out_file = out_files + data.split('.')[0] + '.png'
#
#             clean_sig, sr = librosa.load(data_file)
#
#             # spectrum = librosa.feature.mfcc(clean_sig, sr=sr)    # mfcc
#             # spectrum = librosa.stft(clean_sig)[20:500, :]
#             mel_spec = librosa.feature.melspectrogram(clean_sig, sr=sr, n_fft=512,
#                                                       hop_length=10, win_length=512,
#                                                       window="blackman", n_mels=26)
#             # print(mel_spec.shape)
#             spectrum = librosa.power_to_db(mel_spec)  # 转换为log尺度
#
#
#
#             rgb_image = spectrum_to_rgb(spectrum)
#
#             # 将数组转换为图像对象
#             image = Image.fromarray(rgb_image)
#
#             # 保存为PNG格式
#             image.save(out_file, format="PNG")





clean_sig , sr = librosa.load('./data/O2Miss/O2Miss001.wav')
# spectrum = librosa.feature.mfcc(clean_sig, sr=sr)
spectrogram = librosa.feature.melspectrogram(clean_sig, sr=sr, n_fft=512,
                                                      hop_length=10, win_length=512,
                                                      window="blackman", n_mels=26)
# print(mel_spec.shape)
spectrogram = librosa.power_to_db(spectrogram)  # 转换为log尺度



# 将频谱归一化到0-255的范围
spectrogram_normalized = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram)) * 255

# 将频谱转换为图像
spectrogram_image = Image.fromarray(spectrogram_normalized.astype(np.uint8))

spectrogram_image.show()

to_pil = transforms.ToTensor()

image = to_pil(image)

image = image.numpy()
# 保存图像为PNG文件
# spectrogram_image.save("mel_spectrogram.png")

# Convert spectrum to RGB image
# rgb_image = spectrum_to_rgb(spectrum)

# 将数组转换为图像对象
# image = Image.fromarray(rgb_image)
# image.show()


# image.save("train/output_image.png", format="PNG")



