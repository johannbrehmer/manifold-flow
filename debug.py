import numpy as np
import logging
import sys

logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.INFO,
)
sys.path.append("../")

import torch
from torchvision.datasets import MNIST
from torchvision import transforms

from aef.models.aef import Autoencoder
from aef.trainer import AutoencoderTrainer
from aef.losses import nll, mse

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist = MNIST('./data', download=True, transform=img_transform)

latent_dim = 2
ae = Autoencoder()
ae_trainer = AutoencoderTrainer(ae)

ae_trainer.train(
    dataset=mnist,
    loss_functions=[mse, nll],
    loss_weights=[1., 0.],
    loss_labels=["MSE", "NLL"],
    batch_size=512,
    epochs=20,
)

x = torch.cat([mnist[i][0].unsqueeze(0) for i in range(1000)], dim=0)
y = np.asarray([mnist[i][1] for i in range(1000)])

h = ae.encoder(x)
h = h.detach().numpy().reshape((-1,latent_dim))
print(h)

u = ae.latent(x)
u = u.detach().numpy().reshape((-1,latent_dim))
print(u)
