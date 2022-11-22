# encoding=utf-8
import torch.nn as nn

from utils.mish import Mish


class AutoEncoder(nn.Module):

    def __init__(self, d, k):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d, k),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(k, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, d),
            nn.LeakyReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
