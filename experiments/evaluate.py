#! /usr/bin/env python

import numpy as np
import logging
import sys
import torch
import argparse

sys.path.append("../")

from experiments.inference import mcmc, sq_maximum_mean_discrepancy
from experiments.utils.various import _load_simulator, _create_model, _filename, _create_modelname
from experiments.utils.various import _load_test_samples

logger = logging.getLogger(__name__)


def evaluate_samples(args):
    # Model name
    _create_modelname(args)

    logger.info("Evaluating model %s on generation tasks", args.modelname)

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Data
    simulator = _load_simulator(args)

    if simulator.parameter_dim() is not None:
        logger.info("Data set %s has parameters, skipping generative evaluation", args.dataset)
        return

    # Model
    model = _create_model(args, context_features=simulator.parameter_dim())

    # Generate data
    x_gen = _sample_from_model(args, model)

    # Calculate likelihood of data
    _evaluate_model_samples(args, simulator, x_gen)


def evaluate_inference(args):
    # Model name
    _create_modelname(args)

    logger.info("Evaluating model %s on inference tasks", args.modelname)

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Data
    simulator = _load_simulator(args)

    if simulator.parameter_dim() is None:
        logger.info("Data set %s has no parameters, skipping inference evaluation", args.dataset)
        return

    # Model
    model = _create_model(args, context_features=simulator.parameter_dim())
    model.load_state_dict(torch.load(_filename("model", None, args), map_location=torch.device("cpu")))

    # Evaluate MMD
    model_posterior_samples = _mcmc(simulator, model, n_samples=args.observedsamples)
    np.save(_filename("results", "model_posterior_samples", args), model_posterior_samples)

    true_posterior_samples = _mcmc(simulator, n_samples=args.observedsamples)
    np.save(_filename("results", "true_posterior_samples", args), true_posterior_samples)

    mmd = sq_maximum_mean_discrepancy(model_posterior_samples, true_posterior_samples, scale="ys")
    np.save(_filename("results", "mmd", args), mmd)
    logger.info("MMD between model and true posterior samples: %s", mmd)


def _sample_from_model(args, model):
    logger.info("Sampling from model")
    x_gen = model.sample(n=args.generate).detach().numpy()
    np.save(_filename("results", "samples", args), x_gen)
    return x_gen


def _evaluate_model_samples(args, simulator, x_gen):
    # Likelihood
    logger.info("Calculating likelihood of generated samples")
    log_likelihood_gen = simulator.log_density(x_gen)
    log_likelihood_gen[np.isnan(log_likelihood_gen)] = -1.0e-12
    np.save(_filename("results", "samples_likelihood", args), log_likelihood_gen)

    # Distance from manifold
    try:
        logger.info("Calculating distance from manifold of generated samples")
        distances_gen = simulator.distance_from_manifold(x_gen)
        np.save(_filename("results", "samples_manifold_distance", args), distances_gen)
    except NotImplementedError:
        logger.info("Cannot calculate distance from manifold for dataset %s", args.dataset)


def _mcmc(simulator, model=None, n_samples=1, n_mcmc_samples=1000, slice_sampling=False, step=0.5, thin=1, burnin=100):
    # George's settings: thin = 10, n_mcmc_samples = 5000, burnin = 100

    logger.info(
        "Starting MCMC based on %s after %s observed samples, generating %s posterior samples with %s sampler",
        "true simulator likelihood" if model is None else "neural likelihood estimate",
        n_samples,
        n_mcmc_samples,
        "slice sampler" if slice_sampling else "Metropolis-Hastings sampler",
    )

    # Data
    true_parameters = simulator.default_parameters()
    x_obs = _load_test_samples(args)[:n_samples]
    x_obs_ = torch.tensor(x_obs, dtype=torch.float)

    if model is None:
        # MCMC based on ground truth likelihood
        def log_posterior(params):
            # timer.timer(start="true likelihood")
            log_prob = np.sum(simulator.log_density(x_obs, parameters=params))
            # timer.timer(stop="true likelihood", start="true prior")
            log_prob += simulator.evaluate_log_prior(params)
            # timer.timer(stop="true prior")
            return float(log_prob)

    else:
        # MCMC based on neural likelihood estimator
        def log_posterior(params):
            # timer.timer(start="nde likelihood")
            params_ = np.broadcast_to(params.reshape((-1,  params.shape[-1])), (x_obs.shape[0], params.shape[-1]))
            params_ = torch.tensor(params_, dtype=torch.float)
            log_prob = np.sum(model.log_prob(torch.tensor(x_obs_), context=params_).detach().numpy())
            # timer.timer(stop="nde likelihood", start="nde prior")
            log_prob += simulator.evaluate_log_prior(params)
            # timer.timer(stop="nde prior")
            return float(log_prob)

    if slice_sampling:
        logger.debug("Initializing slice sampler")
        sampler = mcmc.SliceSampler(true_parameters, log_posterior, thin=thin)
    else:
        logger.debug("Initializing Gaussian Metropolis-Hastings sampler")
        sampler = mcmc.GaussianMetropolis(true_parameters, log_posterior, step=step, thin=thin)

    # timer.reset()
    # timer.timer(start="mcmc")

    if burnin > 0:
        logger.info("Starting burn in")
        sampler.gen(burnin)  # burn in
    logger.info("Burn in done, starting main chain")
    posterior_samples = sampler.gen(n_mcmc_samples)
    logger.info("MCMC done")

    # timer.timer(stop="mcmc")
    # timer.report()

    return posterior_samples


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--modelname", type=str, default=None, help="Model name.")
    parser.add_argument("--algorithm", type=str, default="mf", choices=["flow", "pie", "mf"])
    parser.add_argument("--dataset", type=str, default="spherical_gaussian", choices=["spherical_gaussian", "conditional_spherical_gaussian"])

    parser.add_argument("--conditionalouter", action="store_true")
    parser.add_argument("--truelatentdim", type=int, default=8)
    parser.add_argument("--datadim", type=int, default=9)
    parser.add_argument("--epsilon", type=float, default=0.01)

    parser.add_argument("--modellatentdim", type=int, default=8)
    parser.add_argument("--outertransform", type=str, default="affine-coupling")
    parser.add_argument("--innertransform", type=str, default="affine-coupling")
    parser.add_argument("--lineartransform", type=str, default="permutation")
    parser.add_argument("--outerlayers", type=int, default=3)
    parser.add_argument("--innerlayers", type=int, default=5)

    parser.add_argument("--generate", type=int, default=1000)
    parser.add_argument("--observedsamples", type=int, default=10)
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
    evaluate_inference(args)
    logger.info("All done! Have a nice day!")
