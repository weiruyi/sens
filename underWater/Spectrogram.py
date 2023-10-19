import librosa
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import librosa.display
from scipy import signal
from scipy.io import wavfile
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean


def func_format(x, pos):
    return "%d" % (1000 * x)


class Spectrogram:
    """声谱图（语谱图）特征"""
    def __init__(self, input_file, sr=22050, frame_len=512, n_fft=512, win_step=0.01, window="blackman", preemph=0.97, offset=None):
    # def __init__(self, input_file, sr=44100, frame_len=512, n_fft=512, win_step=0.01, window="blackman",preemph=0.97, offset=None):
        """
        初始化
        :param input_file: 输入音频文件
        :param sr: 所输入音频文件的采样率，默认为None
        :param frame_len: 帧长，默认512个采样点(32ms,16kHz),与窗长相同
        :param n_fft: FFT窗口的长度，默认与窗长相同
        :param win_step: 窗移，默认移动2/3，512*2/3=341个采样点(21ms,16kHz)
        :param window: 窗类型，默认汉明窗
        :param preemph: 预加重系数,默认0.97
        """
        self.input_file = input_file
        self.offset = offset
        self.wave_data, self.sr = librosa.load(self.input_file, sr=sr, offset=self.offset, mono=True)  # 音频全部采样点的归一化数组形式数据
        self.wave_data = librosa.effects.preemphasis(self.wave_data, coef=preemph)  # 预加重，系数0.97
        self.window_len = frame_len  # 窗长
        if n_fft is None:
            self.fft_num = self.window_len  # 设置NFFT点数与窗长相等
        else:
            self.fft_num = n_fft
        self.hop_length = round(self.window_len * win_step)  # 重叠部分采样点数设置为窗长的1/3（1/3~1/2）,即帧移(窗移)2/3
        # self.hop_length = 10  # 重叠部分采样点数设置为窗长的1/3（1/3~1/2）,即帧移(窗移)2/3
        self.window = window

    def get_magnitude_spectrogram(self):
        """
        获取幅值谱:fft后取绝对值
        :return: np.ndarray[shape=(1 + n_fft/2, n_frames), dtype=float32]，（257，全部采样点数/(512*2/3)+1）
        """
        # 频谱矩阵：行数=1 + n_fft/2=257，列数=帧数n_frames=全部采样点数/(512*2/3)+1（向上取整）
        # 快速傅里叶变化+汉明窗
        # mag_spec = np.abs(librosa.stft(self.wave_data, n_fft=self.fft_num, hop_length=self.hop_length,
        #                                win_length=self.window_len, window=self.window))
        mag_spec = np.abs(librosa.stft(self.wave_data))
        return mag_spec

    def get_power_spectrogram(self) -> object:
        """
        获取功率谱（能量谱）：幅值谱平方
        :return: np.ndarray[shape=(1 + n_fft/2, n_frames), dtype=float32]，（257，全部采样点数/(512*2/3)+1）
        """
        pow_spec = np.square(self.get_magnitude_spectrogram())
        return pow_spec

    def get_log_power_spectrogram(self):
        """
        获取log尺度功率谱（能量谱）：幅值谱平方S(也就是功率谱),10 * log10(S / ref),其中ref指定为S的最大值
        :return: np.ndarray[shape=(1 + n_fft/2, n_frames), dtype=float32]，（257，全部采样点数/(512*2/3)+1）
        """
        log_pow_spec = librosa.amplitude_to_db(self.get_magnitude_spectrogram(), ref=np.max)  # 转换为log尺度
        return log_pow_spec

    def get_mel_spectrogram(self, n_mels=40):
        """
        获取Mel谱:
        :param n_mels: Mel滤波器组的滤波器数量，默认26
        :return: np.ndarray[shape=(n_mels, n_frames), dtype=float32]，（26，全部采样点数/(512*2/3)+1）
        """
        # 频谱矩阵：行数=n_mels=26，列数=帧数n_frames=全部采样点数/(512*2/3)+1（向上取整）
        # 快速傅里叶变化+汉明窗,Mel滤波器组的滤波器数量 = 26
        mel_spec = librosa.feature.melspectrogram(self.wave_data, sr=self.sr, n_fft=self.fft_num,
                                                  hop_length=self.hop_length, win_length=self.window_len,
                                                  window=self.window, n_mels=n_mels)
        # print(mel_spec.shape)
        log_mel_spec = librosa.power_to_db(mel_spec)  # 转换为log尺度
        return log_mel_spec

    def mag_plot(self, num=None):
        mag_spec = self.get_magnitude_spectrogram()
        librosa.display.specshow(mag_spec, sr=self.sr, hop_length=self.hop_length, x_axis="s", y_axis="linear",
                                 cmap='coolwarm')
        #plt.ylim(0, 9000)
        plt.title("magnitude_spectrogram")
        plt.xlabel("Time/s")
        plt.ylabel("Frequency/Hz")
        # plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(func_format))
        plt.colorbar(shrink=0.7)
        #plt.savefig(r'E:\pycharm\audioanalyze\tmppic\beat.png', dpi=600)

    def pow_plot(self, num=None):
        pow_spec = self.get_power_spectrogram()
        librosa.display.specshow(pow_spec, sr=self.sr, hop_length=self.hop_length, x_axis="s", y_axis="linear",
                                 cmap='coolwarm')
        #plt.ylim(0, 9000)
        plt.title("Power Spectrogram")
        plt.xlabel("Time/s")
        plt.ylabel("Frequency/Hz")
        # plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(func_format))
        plt.colorbar(shrink=0.7)


    def log_pow_plot(self, num=None):
        log_pow_spec = self.get_log_power_spectrogram()
        librosa.display.specshow(log_pow_spec, sr=self.sr, hop_length=1024, x_axis="s", y_axis="log",
                                 cmap='coolwarm')
        plt.ylim(0, 9000)
        plt.title("log_power_spectrogram")
        plt.xlabel("Time/s")
        plt.ylabel("Frequency/Hz")
        # plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(func_format))
        plt.colorbar(shrink=0.7, format="%+02.0f dB")
        #plt.savefig(r'E:\pycharm\audioanalyze\spectrogram_pic\shake\fine\len_2048\up\slice{}'.format(num), dpi=600)
        # plt.savefig(r'E:\pycharm\audioanalyze\spectrogram_pic\shake\lh\fine\len_2048\down\slice{}.png'.format(num), dpi=300)

    def mel_plot(self, num=None, **kwargs):
        mel_spec = self.get_mel_spectrogram(**kwargs)
        # print('shape:' + str(mel_spec.shape))
        # print('hoplength:' + str(self.hop_length))
        librosa.display.specshow(mel_spec, sr=self.sr, hop_length=self.hop_length, x_axis="s", y_axis="mel",
                                 cmap='coolwarm')
        #plt.ylim(0, 9000)
        plt.xlabel("Time/s")
        plt.ylabel("Frequency/Hz")
        plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(func_format))
        plt.title("mel_spectrogram")
        plt.colorbar(shrink=0.7, format="%+02.0f dB")
        # plt.savefig(r'E:\pycharm\audioanalyze\a'.format(num), format="svg")

    def plot(self, fig=None, show=True, num=None, **kwargs):

        """
        绘制声谱图
        :param num:
        :param fig: 指定绘制何种声谱图，mag/pow/log_pow/mel,默认都绘制
        :param show: 默认最后调用plt.show()，显示图形
        :return: None
        """
        self.num = num

        if fig == "mag":
            self.mag_plot(num=self.num)
        elif fig == "pow":
            self.pow_plot(num=self.num)
        elif fig == "log_pow":
            self.log_pow_plot(num=self.num)
        elif fig == "mel":
            self.mel_plot(**kwargs)

        else:
            plt.figure(figsize=(16, 8))
            plt.subplot(2, 2, 1)
            self.mag_plot()

            plt.subplot(2, 2, 2)
            self.pow_plot()

            plt.subplot(2, 2, 3)
            self.log_pow_plot()

            plt.subplot(2, 2, 4)
            self.mel_plot(**kwargs)

        plt.tight_layout()
        if show:
            plt.show()


class Similarity:
    def __init__(self, spec_1, spec_2):
        self.spec_1 = spec_1
        self.spec_2 = spec_2

    def cal_sim(self,sig_1, sig_2):
        similarity = cosine_similarity(sig_1, sig_2)  # 余弦相似度

        return similarity


    def plot_sim(self, similarity):
        plt.imshow(similarity, cmap='hot', origin='lower')
        plt.colorbar()
        plt.xlabel('Audio 2 Frame')
        plt.ylabel('Audio 1 Frame')
        plt.title('Similarity Matrix')
    def mag_plot(self):
        mag_1 = self.spec_1.get_magnitude_spectrogram()
        mag_2 = self.spec_2.get_magnitude_spectrogram()
        mag_1 = mag_1.T
        mag_2 = mag_2.T
        similarity = self.cal_sim(mag_1, mag_2)
        self.plot_sim(similarity)


    def pow_plot(self):
        pow_1 = self.spec_1.get_power_spectrogram().T
        pow_2 = self.spec_2.get_power_spectrogram().T
        similarity = self.cal_sim(pow_1, pow_2)
        self.plot_sim(similarity)

    def log_pow_plot(self):
        log_pow_1 = self.spec_1.get_log_power_spectrogram().T
        log_pow_2 = self.spec_2.get_log_power_spectrogram().T
        similarity = self.cal_sim(log_pow_1, log_pow_2)
        self.plot_sim(similarity)


    def mel_plot(self):
        mel_1 = self.spec_1.get_mel_spectrogram(n_mels=26).T
        mel_2 = self.spec_2.get_mel_spectrogram(n_mels=26).T
        similarity = self.cal_sim(mel_1, mel_2)
        self.plot_sim(similarity)

    def plot(self, fig=None, show=True):
        if fig == "mag":
            self.mag_plot()
        elif fig == "pow":
            self.pow_plot()
        elif fig == "log_pow":
            self.log_pow_plot()
        elif fig == "mel":
            self.mel_plot()
        else:
            plt.figure(figsize=(16, 8))
            plt.subplot(2, 2, 1)
            self.mag_plot()
            plt.subplot(2, 2, 2)
            self.pow_plot()
            plt.subplot(2, 2, 3)
            self.log_pow_plot()
            plt.subplot(2, 2, 4)
            self.mel_plot()
        plt.tight_layout()
        if show:
            plt.show()


if __name__ == "__main__":
    spectrogram_f = Spectrogram('../dataset/prior/wry/帮助我.wav')
    # spectrogram_f.plot(fig=" ")


    spec_2 = Spectrogram('../dataset/prior/qc/帮助我.wav')

    sim = Similarity(spectrogram_f, spec_2)
    sim.plot(fig='')

    #
    # # spectrogram_f = Spectrogram('./data_a/fyp/fypleft/fypleft002.wav')
    # # # spectrogram_f.plot(fig="")
    # #
    # #
    # # spec_2 = Spectrogram('./data_a/fyp/fypleft/fypleft003.wav')
    # #
    # # sim = Similarity(spectrogram_f, spec_2)
    # # sim.plot(fig='')