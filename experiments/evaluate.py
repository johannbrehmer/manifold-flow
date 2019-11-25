#! /usr/bin/env python

import numpy as np
import logging
import sys
import torch
import argparse

sys.path.append("../")

from experiments.inference import mcmc, sq_maximum_mean_discrepancy
from experiments.utils import _load_simulator, _load_model, _filename, _create_modelname, _load_training_dataset
from experiments.utils import _load_test_samples

logger = logging.getLogger(__name__)


def evaluate_samples(args):
    # Model name
    _create_modelname(args)

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
    # Model name
    _create_modelname(args)

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Data
    simulator = _load_simulator(args)

    # Model
    model = _load_model(args)

    # Evaluate MMD
    mmds = []
    for samples in args.samplesizes:
        true_posterior_samples = _mcmc(simulator, n_samples=samples)
        model_posterior_samples = _mcmc(simulator, model, n_samples=samples)
        mmds.append(sq_maximum_mean_discrepancy(model_posterior_samples, true_posterior_samples, scale="ys"))
    np.save(_filename("results", "mmd", args), mmds)


def _sample_from_model(args, model):
    logger.info("Sampling from model")
    x_gen = model.sample(n=args.generate).detach().numpy()
    np.save(_filename("results", "samples", args), x_gen)
    return x_gen


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


def _mcmc(simulator, model=None, thin=10, n_samples=100, n_mcmc_samples=5000, burnin=100):
    # Data
    true_parameters = simulator.default_parameters()
    x_obs = _load_test_samples(args)[:n_samples]

    if model is None:
        # MCMC based on ground truth likelihood
        def log_posterior(params):
            log_prob = np.sum(simulator.log_density(x_obs, parameter=params))
            log_prob += simulator.evaluate_log_prior(params)
            return log_prob
    else:
        # MCMC based on neural likelihood estimator
        def log_posterior(params):
            log_prob = np.sum(model.log_prob(torch.array(x_obs), context=torch.array(params)).detach().numpy())
            log_prob += simulator.evaluate_log_prior(params)
            return log_prob

    sampler = mcmc.SliceSampler(true_parameters, log_posterior, thin=thin)
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

    parser.add_argument("--generate", type=int, default=1000)
    parser.add_argument("--samplesizes", nargs="+", type=int, default=[1,2,5,10,20,50,100,200,500,1000])
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
