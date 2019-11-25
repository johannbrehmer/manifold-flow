#! /usr/bin/env python

import numpy as np
import logging
import sys
import torch
import argparse
import os

sys.path.append("../")

from manifold_flow.inference import mcmc
from experiments.utils import _load_simulator, _load_model, _filename

logger = logging.getLogger(__name__)


def evaluate_samples(args):
    if args.modelname is None:
        args.modelname = "{}_{}_{}_{}_{}_{:.3f}".format(args.algorithm, args.modellatentdim, args.dataset, args.truelatentdim, args.datadim, args.epsilon)
    logger.info(
        "Evaluating generative mode of model %s on data set %s (data dim %s, true latent dim %s)",
        args.modelname,
        args.dataset,
        args.datadim,
        args.truelatentdim,
    )

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Data
    simulator = _load_simulator(args)

    # Model
    model = _load_model(args)

    # Generate data
    x_gen = _sample_from_model(args, model)

    # Calculate likelihood of data
    _evaluate_model_samples(args, simulator, x_gen)


def evaluate_inference(args):
    if args.modelname is None:
        args.modelname = "{}_{}_{}_{}_{}_{:.3f}".format(args.algorithm, args.modellatentdim, args.dataset, args.truelatentdim, args.datadim, args.epsilon)
    logger.info(
        "Evaluating inference for model %s on data set %s (data dim %s, true latent dim %s)",
        args.modelname,
        args.dataset,
        args.datadim,
        args.truelatentdim,
    )

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Data
    simulator = _load_simulator(args)

    # Model
    model = _load_model(args)

    # TODO: MCMC, MMD calculation


def _evaluate_model_samples(args, simulator, x_gen):
    # Likelihood
    logger.info("Calculating likelihood of generated samples")
    log_likelihood_gen = simulator.log_density(x_gen)
    log_likelihood_gen[np.isnan(log_likelihood_gen)] = -1.e-12
    np.save(_filename("results", "samples_likelihood", args), log_likelihood_gen)

    # Distance from manifold
    try:
        logger.info("Calculating distance from manifold of generated samples")
        distances_gen = simulator.distance_from_manifold(x_gen)
        np.save(_filename("results", "samples_manifold_distance", args), distances_gen)
    except NotImplementedError:
        logger.info("Cannot calculate distance from manifold for dataset %s", args.dataset)


def _sample_from_model(args, model):
    logger.info("Sampling from model")
    x_gen = model.sample(n=args.samples).detach().numpy()
    np.save(_filename("results", "samples", args), x_gen)
    return x_gen


def _mcmc(simulator, model, thin=10, n_samples=100, n_mcmc_samples=5000, burnin=100):
    parameters = simulator.default_parameters()
    x = simulator.sample(parameters=parameters, n=n_samples)
    prior = simulator.prior()

    log_posterior = lambda t: model.log_prob(torch.array(x), context=torch.array(t)) + prior.eval(t)
    sampler = mcmc.SliceSampler(parameters, log_posterior, thin=thin)
    sampler.gen(burnin)  # burn in
    posterior_samples = sampler.gen(n_mcmc_samples)

    return posterior_samples


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

    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--dir", type=str, default="../")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO
    )
    logger.info("Hi!")
    evaluate_samples(args)
    logger.info("All done! Have a nice day!")
