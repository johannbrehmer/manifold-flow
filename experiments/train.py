#! /usr/bin/env python

import numpy as np
import logging
import sys
import torch
import argparse
from torch import optim

sys.path.append("../")

from manifold_flow.training import ManifoldFlowTrainer, losses
from experiments.utils import _load_model, _filename, _load_training_data

logger = logging.getLogger(__name__)


def train(args):
    if args.modelname is None:
        args.modelname = "{}_{}_{}_{}_{}_{:.3f}".format(args.algorithm, args.modellatentdim, args.dataset, args.truelatentdim, args.datadim, args.epsilon)
    logger.info(
        "Training model %s algorithm %s and %s latent dims on data set %s (data dim %s, true latent dim %s)",
        args.modelname,
        args.algorithm,
        args.modellatentdim,
        args.dataset,
        args.datadim,
        args.truelatentdim,
    )

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Data
    dataset = _load_training_data(args)

    # Model
    model = _load_model(args)

    # Train
    trainer = ManifoldFlowTrainer(model)

    if args.algorithm in ["flow", "pie"]:
        logger.info("Starting training on NLL")
        learning_curves = trainer.train(
            dataset=dataset,
            loss_functions=[losses.nll],
            loss_labels=["NLL"],
            loss_weights=[1.0],
            batch_size=args.batchsize,
            epochs=args.epochs,
            initial_lr=args.lr,
            scheduler=optim.lr_scheduler.CosineAnnealingLR,
        )
        learning_curves = np.vstack(learning_curves).T
    else:
        logger.info("Starting training on MSE")
        learning_curves = trainer.train(
            dataset=dataset,
            loss_functions=[losses.mse, losses.nll],
            loss_labels=["MSE", "NLL"],
            loss_weights=[args.alpha, args.beta],
            batch_size=args.batchsize,
            epochs=args.epochs // 2,
            initial_lr=args.lr,
            scheduler=optim.lr_scheduler.CosineAnnealingLR,
        )
        learning_curves = np.vstack(learning_curves).T

        logger.info("Starting training on NLL")
        learning_curves2 = trainer.train(
            dataset=dataset,
            loss_functions=[losses.mse, losses.nll],
            loss_labels=["MSE", "NLL"],
            loss_weights=[0.0, 1.0],
            batch_size=args.batchsize,
            epochs=args.epochs // 2,
            initial_lr=args.lr,
            scheduler=optim.lr_scheduler.CosineAnnealingLR,
        )

        learning_curves2 = np.vstack(learning_curves2).T
        learning_curves = np.vstack((learning_curves, learning_curves2))

    # Save
    logger.info("Saving model")
    torch.save(model.state_dict(), _filename("model", None, args))
    np.save(_filename("learning_curve", None, args), learning_curves)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--modelname", type=str, default=None, help="Model name.")
    parser.add_argument("--algorithm", type=str, default="mf", choices=["flow", "pie", "mf"])
    parser.add_argument("--dataset", type=str, default="spherical_gaussian", choices=["spherical_gaussian"])

    parser.add_argument("--truelatentdim", type=int, default=10)
    parser.add_argument("--datadim", type=int, default=15)
    parser.add_argument("--epsilon", type=float, default=0.01)

    parser.add_argument("--modellatentdim", type=int, default=10)
    parser.add_argument("--transform", type=str, default="affine-coupling")
    parser.add_argument("--outerlayers", type=int, default=5)
    parser.add_argument("--innerlayers", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--beta", type=float, default=1.0e-2)

    parser.add_argument("--dir", type=str, default="../")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO
    )
    logger.info("Hi!")
    train(args)
    logger.info("All done! Have a nice day!")
