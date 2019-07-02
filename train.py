#! /usr/bin/env python

import numpy as np
import logging
import sys
import torch
import argparse

sys.path.append("../")

from aef.models.autoencoding_flow import TwoStepAutoencodingFlow
from aef.trainer import AutoencodingFlowTrainer, NumpyDataset
from aef.losses import nll, mse

logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.INFO,
)


def train(
    model_filename,
    dataset="tth",
    latent_dim=10,
    flow_steps_inner=5,
    flow_steps_outer=5,
    batch_size=128,
    epochs=(20, 20),
    alpha=1.0e-3,
    lr=(1.0e-4, 1.0e-6),
    base_dir=".",
):
    # Data
    if dataset == "tth":
        x = np.load("{}/data/tth/x_train.npy".format(base_dir))
        x_means = np.mean(x, axis=0)
        x_stds = np.std(x, axis=0)
        x = (x - x_means[np.newaxis, :]) / x_stds[np.newaxis, :]
        y = np.ones(x.shape[0])
        data = NumpyDataset(x, y)
        data_dim = 48
    else:
        raise NotImplementedError("Unknown dataset {}".format(dataset))

    # Model
    ae = TwoStepAutoencodingFlow(
        data_dim=data_dim,
        latent_dim=latent_dim,
        steps_inner=flow_steps_inner,
        steps_outer=flow_steps_outer,
    )

    # Trainer
    trainer = AutoencodingFlowTrainer(ae, double_precision=True)

    # Train
    trainer.train(
        optimizer=torch.optim.Adam,
        dataset=data,
        loss_functions=[mse],
        loss_labels=["MSE"],
        loss_weights=[1.0],
        batch_size=batch_size,
        epochs=epochs[0],
        verbose="all",
        initial_lr=lr[0],
        final_lr=lr[1],
    )
    trainer.train(
        optimizer=torch.optim.Adam,
        dataset=data,
        loss_functions=[mse, nll],
        loss_labels=["MSE", "NLL"],
        loss_weights=[1.0, alpha],
        batch_size=batch_size,
        epochs=epochs[1],
        verbose="all",
        initial_lr=lr[0],
        final_lr=lr[1],
        parameters=ae.outer_transform.parameters(),
    )

    # Save
    torch.save(ae.state_dict(), "{}/data/models/{}.pt".format(base_dir, model_filename))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strong lensing experiments: simulation"
    )
    parser.add_argument("name", type=str, help="Model name.")
    parser.add_argument("--dataset", type=str, default="tth", choices=["tth"])
    parser.add_argument("--latent", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--lrdecay", type=float, default=1.0e-2)
    parser.add_argument("--dir", type=str, default=".")
    return parser.parse_args()


if __name__ == "__main__":
    logging.info("Hi!")

    args = parse_args()

    train(
        model_filename=args.name,
        dataset=args.dataset,
        latent_dim=args.latent,
        flow_steps_inner=args.steps,
        flow_steps_outer=args.steps,
        batch_size=args.batchsize,
        epochs=(args.epochs // 2, args.epochs // 2),
        alpha=args.alpha,
        lr=(args.lr, args.lr * args.lrdecay),
        base_dir=args.dir,
    )

    logging.info("All done! Have a nice day!")
