#! /usr/bin/env python

import numpy as np
import logging
import sys
import torch
import argparse
import os
from torch import optim

sys.path.append("../")

from manifold_flow.flows import ManifoldFlow, Flow, PIE
from manifold_flow.training import ManifoldFlowTrainer, losses, NumpyDataset
from experiments.data_generation import SphericalGaussianSimulator

logger = logging.getLogger(__name__)


def evaluate(args):
    if args.modelname is None:
        args.modelname = "{}_{}_{}_{}_{}_{:.3f}".format(args.algorithm, args.modellatentdim, args.dataset, args.truelatentdim, args.datadim, args.epsilon)
    logger.info(
        "Evaluating model %s on data set %s (data dim %s, true latent dim %s)",
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
    if args.dataset == "spherical_gaussian":
        simulator = SphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    else:
        raise NotImplementedError("Unknown dataset {}".format(args.dataset))

    # Model
    if args.algorithm == "flow":
        logger.info("Loading standard flow with %s layers", args.outerlayers)
        model = Flow(data_dim=args.datadim, steps=args.outerlayers, transform=args.transform)
    elif args.algorithm == "pie":
        logger.info("Loading PIE with %s latent dimensions and %s + %s layers", args.modellatentdim, args.outerlayers, args.innerlayers)
        model = PIE(
            data_dim=args.datadim,
            latent_dim=args.modellatentdim,
            steps_inner=args.innerlayers,
            steps_outer=args.outerlayers,
            outer_transform=args.transform,
            inner_transform=args.transform,
        )
    elif args.algorithm == "mf":
        logger.info("Loading manifold flow with %s latent dimensions and %s + %s layers", args.modellatentdim, args.outerlayers, args.innerlayers)
        model = ManifoldFlow(
            data_dim=args.datadim,
            latent_dim=args.modellatentdim,
            steps_inner=args.innerlayers,
            steps_outer=args.outerlayers,
            outer_transform=args.transform,
            inner_transform=args.transform,
        )
    else:
        raise NotImplementedError("Unknown algorithm {}".format(args.algorithm))
    model.load_state_dict(torch.load("{}/experiments/data/models/{}.pt".format(args.dir, args.modelname)))

    # Generate data
    logger.info("Sampling from model")
    x_gen = model.sample(n=args.samples).detach().numpy()

    logger.info("Saving generated samples to %s", "{}/experiments/data/results/{}_samples.npy".format(args.dir, args.modelname))
    os.makedirs("{}/experiments/data/results".format(args.dir, args.modelname), exist_ok=True)
    np.save("{}/experiments/data/results/{}_samples.npy".format(args.dir, args.modelname), x_gen)

    # Calculate likelihood of data
    logger.info("Calculating likelihood of generated samples")
    log_likelihood_gen = simulator.log_density(x_gen)

    logger.info("Saving likelihood to %s", "{}/experiments/data/results/{}_samples_likelihood.npy".format(args.dir, args.modelname))
    np.save("{}/experiments/data/results/{}_samples_likelihood.npy".format(args.dir, args.modelname), log_likelihood_gen)

    # Distance from manifold
    try:
        logger.info("Calculating distance from manifold of generated samples")
        distances_gen = simulator.distance_from_manifold(x_gen)

        logger.info("Saving likelihood to %s", "{}/experiments/data/results/{}_samples_manifold_distance.npy".format(args.dir, args.modelname))
        np.save("{}/experiments/data/results/{}_samples_manifold_distance.npy".format(args.dir, args.modelname), distances_gen)

    except NotImplementedError:
        logger.info("Cannot calculate distance from manifold for dataset %s", args.dataset)


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

    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--logpmin", type=float, default=-1000.)
    parser.add_argument("--dir", type=str, default="../")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO
    )
    logger.info("Hi!")
    evaluate(args)
    logger.info("All done! Have a nice day!")
