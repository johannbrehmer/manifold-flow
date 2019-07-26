#! /usr/bin/env python

import numpy as np
import logging
import sys
import torch
import argparse

sys.path.append("../")

from aef.models.autoencoding_flow import Flow, TwoStepAutoencodingFlow
from aef.trainer import AutoencodingFlowTrainer, NumpyDataset
from aef.losses import nll, mse
from nsf.experiments import images_data

logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.INFO,
)


def train(
    model_filename,
    dataset="tth",
    data_dim=None,
    latent_dim=10,
    flow_steps_inner=5,
    flow_steps_outer=5,
    batch_size=128,
    epochs=(20, 20),
    alpha=1.0e-3,
    lr=(1.0e-4, 1.0e-6),
    base_dir=".",
):
    logging.info("Starting training of model %s on data set %s", model_filename, dataset)

    # Data
    if dataset == "tth":
        x = np.load("{}/data/tth/x_train.npy".format(base_dir))
        x_means = np.mean(x, axis=0)
        x_stds = np.std(x, axis=0)
        x = (x - x_means[np.newaxis, :]) / x_stds[np.newaxis, :]
        y = np.ones(x.shape[0])
        data = NumpyDataset(x, y)
        data_dim = 48
        mode = "vector"
        logging.info("Loaded tth data with %s-dimensional data", data_dim)

    elif dataset == "gaussian":
        assert data_dim is not None
        x = np.load("{}/data/gaussian/gaussian_8_{}_x_train.npy".format(base_dir, data_dim))
        y = np.ones(x.shape[0])
        data = NumpyDataset(x, y)
        mode = "vector"
        logging.info("Loaded linear Gaussian data with %s-dimensional data", data_dim)

    elif dataset == "spherical_gaussian":
        assert data_dim is not None
        x = np.load("{}/data/spherical_gaussian/spherical_gaussian_15_{}_x_train.npy".format(base_dir, data_dim))
        y = np.ones(x.shape[0])
        data = NumpyDataset(x, y)
        mode = "vector"
        logging.info("Loaded spherical Gaussian data with %s-dimensional data", data_dim)

    elif dataset == "imagenet":
        dataset = aef_data.get_data('imagenet-64-fast', 8, train=True, valid_frac=0.)
        mode = "image"
        logging.info("Loaded imagenet data with %s-dimensional data", data_dim)

    else:
        raise NotImplementedError("Unknown dataset {}".format(dataset))

    # Stop simulations where latent dim is larger than x dim
    if isinstance(data_dim, int) and latent_dim > data_dim:
        logging.info("Latent dim is larger than data dim, skipping this")
        return

    # Model
    if latent_dim is None:
        logging.info("Creating plain flow")
        model = Flow(
            data_dim=data_dim,
            steps=flow_steps_outer,
            mode=mode,
        )

    else:
        logging.info("Creating auto-encoding flow with %s latent dimensions")
        model = TwoStepAutoencodingFlow(
            data_dim=data_dim,
            latent_dim=latent_dim,
            steps_inner=flow_steps_inner,
            steps_outer=flow_steps_outer,
            mode=mode,
        )

    # Trainer
    trainer = AutoencodingFlowTrainer(model, double_precision=True)

    # Train
    if latent_dim is None:
        logging.info("Starting training on NLL")
        trainer.train(
            optimizer=torch.optim.Adam,
            dataset=data,
            loss_functions=[nll],
            loss_labels=["NLL"],
            loss_weights=[1.0],
            batch_size=batch_size,
            epochs=epochs,
            verbose="all",
            initial_lr=lr[0],
            final_lr=lr[1],
        )
    else:
        logging.info("Starting training on MSE")
        trainer.train(
            optimizer=torch.optim.Adam,
            dataset=data,
            loss_functions=[mse],
            loss_labels=["MSE"],
            loss_weights=[1.0],
            batch_size=batch_size,
            epochs=epochs // 2,
            verbose="all",
            initial_lr=lr[0],
            final_lr=lr[1],
        )
        logging.info("Starting training on MSE and NLL")
        trainer.train(
            optimizer=torch.optim.Adam,
            dataset=data,
            loss_functions=[mse, nll],
            loss_labels=["MSE", "NLL"],
            loss_weights=[1.0, alpha],
            batch_size=batch_size,
            epochs=epochs - epochs // 2,
            verbose="all",
            initial_lr=lr[0],
            final_lr=lr[1],
            parameters=model.outer_transform.parameters(),
        )

    # Save
    logging.info("Saving model to %s", "{}/data/models/{}.pt".format(base_dir, model_filename))
    torch.save(model.state_dict(), "{}/data/models/{}.pt".format(base_dir, model_filename))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strong lensing experiments: simulation"
    )
    parser.add_argument("name", type=str, help="Model name.")
    parser.add_argument("--dataset", type=str, default="tth", choices=["tth", "gaussian", "spherical_gaussian"])
    parser.add_argument("-x", type=int, default=None)
    parser.add_argument("--latent", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--lrdecay", type=float, default=0.1)
    parser.add_argument("--dir", type=str, default=".")
    return parser.parse_args()


if __name__ == "__main__":
    logging.info("Hi!")

    args = parse_args()

    train(
        model_filename=args.name,
        dataset=args.dataset,
        data_dim=args.x,
        latent_dim=args.latent,
        flow_steps_inner=args.steps,
        flow_steps_outer=args.steps,
        batch_size=args.batchsize,
        epochs=args.epochs,
        alpha=args.alpha,
        lr=(args.lr, args.lr * args.lrdecay),
        base_dir=args.dir,
    )

    logging.info("All done! Have a nice day!")
