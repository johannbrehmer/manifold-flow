import numpy as np
import logging
import sys

sys.path.append("../")

from train import train

logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.DEBUG,
)

logging.info("Hi!")

train(
    "debug",
    dataset="cifar",
    latent_dim=32,
    flow_steps_outer=3,
    batch_size=64,
    epochs=10,
    alpha=1.0e-3,
    lr=(1.0e-3, 1.0e-5),
    base_dir=".",
)
