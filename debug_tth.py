import numpy as np
import logging
import sys

sys.path.append("../")

from aef.models.autoencoding_flow import TwoStepAutoencodingFlow
from aef.trainer import AutoencodingFlowTrainer, NumpyDataset
from aef.losses import nll, mse
from train import train

logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.DEBUG,
)

logging.info("Hi!")

train(
    "debug",
    dataset="tth",
    latent_dim=10,
    flow_steps_inner=5,
    flow_steps_outer=5,
    batch_size=128,
    epochs=(20, 20),
    alpha=1.0e-3,
    lr=(1.0e-4, 1.0e-6),
    base_dir="."
)
