import numpy as np
import logging
import sys

sys.path.append("../")

from aef.models.autoencoding_flow import TwoStepAutoencodingFlow
from aef.trainer import AutoencodingFlowTrainer, NumpyDataset
from aef.losses import nll, mse

logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.DEBUG,
)

logging.info("Hi!")

x = np.load("data/tth/x_train.npy")
logging.info("Data shape: %s", x.shape)
y = np.ones(x.shape[0])
tth_data = NumpyDataset(x, y)

ae = TwoStepAutoencodingFlow(data_dim=48, latent_dim=10, steps_inner=5, steps_outer=5)

trainer = AutoencodingFlowTrainer(ae)
trainer.train(
    dataset=tth_data,
    loss_functions=[mse],
    loss_labels=["MSE"],
    loss_weights=[1.0],
    batch_size=256,
    epochs=5,
    verbose="all",
    initial_lr=1.0e-3,
    final_lr=1.0e-4,
)
