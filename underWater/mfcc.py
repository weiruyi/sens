import librosa
from matplotlib import pyplot as plt
import librosa.display
from sklearn.metrics.pairwise import cosine_similarity

class MFCC:
    def __init__(self, input_file, sr=None, class_name=''):
        self.input_file = input_file
        self.class_name = class_name
        self.clean_sig , self.sr = librosa.load(self.input_file, sr=sr)
        self.mfcc = librosa.feature.mfcc(self.clean_sig, sr=self.sr)

    def get_mfcc_spec(self):
        return self.mfcc

    def plot_mfcc(self, save=None):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(self.mfcc, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC')
        plt.xlabel('Time')
        plt.ylabel('MFCC Coefficients')
        plt.tight_layout()
        if save:
            path = self.input_file.split('/')[-1].split('.')[0]
            plt.savefig('D:/hnu/python/DenseNet/data/train/' + self.class_name + '/'+ path + '.png')
        plt.show()

    def cal_sim(self, mfcc_2):
        mfcc_2_spec = mfcc_2.get_mfcc_spec()
        similarity = cosine_similarity(self.mfcc.T, mfcc_2_spec.T)  # 余弦相似度
        return similarity

    def plot_sim(self, similarity):
        plt.imshow(similarity, cmap='hot', origin='lower')
        plt.colorbar()
        plt.xlabel('Audio 2 Frame')
        plt.ylabel('Audio 1 Frame')
        plt.title('MFCC Similarity Matrix')
        plt.show()


if __name__ == "__main__":
    mfcc_1 = MFCC('../m4aFiles/掉头1.wav', class_name='Up')
    # mfcc_2 = MFCC('./data/fyp/fypturn/fypturn001.wav')

    mfcc_spec = mfcc_1.get_mfcc_spec()

    # print(mfcc_spec.shape)

    mfcc_1.plot_mfcc(save=False)

    # mfcc_1_spec = mfcc_1.get_mfcc_spec()
    # sim = mfcc_1.cal_sim(mfcc_2)
    # mfcc_1.plot_sim(sim)


