import numpy as np
import logging
import sys
import torch
from torchvision.datasets import MNIST
from torchvision import transforms

sys.path.append("../")
from aef.models.autoencoding_flow import TwoStepAutoencodingFlow
from aef.trainer import AutoencodingFlowTrainer
from aef.losses import nll, mse

logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.INFO,
)
logging.info("Hi!")

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist = MNIST('./data', download=True, transform=img_transform)

ae = TwoStepAutoencodingFlow(data_dim=28*28, latent_dim=16)

trainer = AutoencodingFlowTrainer(ae, output_filename="output/aef_phase1")
trainer.train(
    dataset=mnist,
    loss_functions=[mse, nll],
    loss_labels=["MSE", "NLL"],
    loss_weights=[1., 1.],
    batch_size=256,
    epochs=10,
    verbose="all",
    initial_lr=1.e-3,
    final_lr=1.e-5
)
