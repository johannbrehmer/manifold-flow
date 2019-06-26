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
    level=logging.INFO,
)

logging.info("Hi!")

x = np.load("./data/tth/x_test.npy")
tth_data = NumpyDataset(x, x)

ae = TwoStepAutoencodingFlow(data_dim=48, latent_dim=8, n_mades_inner=3, n_mades_outer=1)

trainer = AutoencodingFlowTrainer(ae, output_filename="output/aef_phase1")
trainer.train(
    dataset=tth_data,
    loss_functions=[mse],
    loss_labels=["MSE"],
    loss_weights=[1.],
    batch_size=256,
    epochs=5,
    verbose="all",
    initial_lr=1.e-4,
    final_lr=1.e-5,
)
