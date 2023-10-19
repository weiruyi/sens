import torch.nn as nn


class classifyDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.TDNN = nn.Sequential(
            nn.Conv1d(1024, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d()
        )
