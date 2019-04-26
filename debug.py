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

from aef.models.ae_latent_flow import ConvolutionalAutoencoder
from aef.trainer import AutoencoderTrainer
from aef.losses import nll, mse

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist = MNIST('./data', download=True, transform=img_transform)

ae = ConvolutionalAutoencoder()
ae_trainer = AutoencoderTrainer(ae)

logging.info("Hi!")

ae_trainer.train(
    dataset=mnist,
    loss_functions=[mse, nll],
    loss_weights=[1., 0.1],
    loss_labels=["MSE", "NLL"],
    batch_size=256,
    epochs=20,
    verbose="all",
)

logging.info("Done")
