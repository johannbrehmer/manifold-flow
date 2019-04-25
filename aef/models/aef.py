import torch
import torchvision
from torch import nn
from aef.models.flows import FlowSequential, MADE, BatchNormFlow, Reverse


class Autoencoder(nn.Module):
    def __init__(self, n_mades=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

        modules = []
        for _ in range(n_mades):
            modules += [
                MADE(8 * 2 * 2, 100, None, act='relu'),
                BatchNormFlow(8 * 2 * 2),
                Reverse(8 * 2 * 2)
            ]
        self.flow=FlowSequential(*modules)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

    def log_prob(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        log_prob = self.flow.log_probs(z)
        return log_prob

    def sample(self, u=None, n=1):
        z = self.flow.sample(noise=u, num_samples=n)
        z = z.view(z.size(0), 8, 2, 2)
        x = self.decoder(z)
        return x
