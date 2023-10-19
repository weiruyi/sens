import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# 加载两个音频文件，并提取其频谱特征
audio1, sr1 = librosa.load('../underWater/data/airleft002.wav')
audio2, sr2 = librosa.load('../underWater/data/airleft001.wav')

# 计算音频的频谱特征
spectrogram1 = librosa.feature.melspectrogram(audio1, sr=sr1)
spectrogram2 = librosa.feature.melspectrogram(audio2, sr=sr2)

# 计算频谱之间的DTW距离
distance, _ = fastdtw(spectrogram1.T, spectrogram2.T, dist=euclidean)

print("DTW距离:", distance)