#! /usr/bin/env python

import numpy as np
import logging
import sys
import torch
import argparse

sys.path.append("../")

from experiments.inference import mcmc, sq_maximum_mean_discrepancy
from experiments.utils.loading import load_simulator, load_test_samples
from experiments.utils.names import create_filename, create_modelname, ALGORITHMS, SIMULATORS
from experiments.utils.models import create_model
from experiments.simulators.base import IntractableLikelihoodError

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # What what what
    parser.add_argument("--modelname", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default="mf", choices=ALGORITHMS)
    parser.add_argument("--dataset", type=str, default="spherical_gaussian", choices=SIMULATORS)

    # Dataset details
    parser.add_argument("--truelatentdim", type=int, default=2)
    parser.add_argument("--datadim", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.01)

    # Model details
    parser.add_argument("--modellatentdim", type=int, default=2)
    parser.add_argument("--specified", action="store_true")
    parser.add_argument("--outertransform", type=str, default="affine-coupling")
    parser.add_argument("--innertransform", type=str, default="affine-coupling")
    parser.add_argument("--lineartransform", type=str, default="permutation")
    parser.add_argument("--outerlayers", type=int, default=4)
    parser.add_argument("--innerlayers", type=int, default=8)
    parser.add_argument("--conditionalouter", action="store_true")
    parser.add_argument("--outercouplingmlp", action="store_true")
    parser.add_argument("--outercouplinglayers", type=int, default=3)
    parser.add_argument("--outercouplinghidden", type=int, default=256)

    # Evaluation settings
    parser.add_argument("--generate", type=int, default=10000)
    parser.add_argument("--observedsamples", type=int, default=10)
    parser.add_argument("--slicesampler", action="store_true")
    parser.add_argument("--mcmcstep", type=float, default=0.2)
    parser.add_argument("--thin", type=int, default=10)

    # Other settings
    parser.add_argument("--dir", type=str, default="../")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skipmodelmcmc", action="store_true")
    parser.add_argument("--skiptruthmcmc", action="store_true")

    return parser.parse_args()


def _sample_from_model(args, model):
    logger.info("Sampling from model")
    x_gen = model.sample(n=args.generate).detach().numpy()
    np.save(create_filename("results", "samples", args), x_gen)
    return x_gen


def _evaluate_model_samples(args, simulator, x_gen):
    # Likelihood
    logger.info("Calculating likelihood of generated samples")
    log_likelihood_gen = simulator.log_density(x_gen)
    log_likelihood_gen[np.isnan(log_likelihood_gen)] = -1.0e-12
    np.save(create_filename("results", "samples_likelihood", args), log_likelihood_gen)

    # Distance from manifold
    try:
        logger.info("Calculating distance from manifold of generated samples")
        distances_gen = simulator.distance_from_manifold(x_gen)
        np.save(create_filename("results", "samples_manifold_distance", args), distances_gen)
    except NotImplementedError:
        logger.info("Cannot calculate distance from manifold for dataset %s", args.dataset)


def _evaluate_test_samples(args, simulator, model=None, samples=1000, batchsize=100):

    if model is None:
        logger.info("Evaluating true log likelihood of test samples")
    else:
        logger.info("Evaluating model likelihood of test samples")

    x = load_test_samples(args)[:samples]

    if model is None:
        if simulator.parameter_dim() is None:
            params = None
        else:
            params = torch.tensor([simulator.default_parameters() for _ in x], dtype=torch.float)
        log_prob = simulator.log_density(x, parameters=params)
        reco_error = np.zeros(x.shape[0])

    else:
        log_prob = []
        reco_error = []
        n_batches = (samples - 1) // batchsize + 1

        for i in range(n_batches):
            logger.debug("Evaluating batch %s / %s", i + 1, n_batches)

            x_ = torch.tensor(x[i * batchsize : (i + 1) * batchsize], dtype=torch.float)
            if simulator.parameter_dim() is None:
                params = None
            else:
                params = torch.tensor([simulator.default_parameters() for _ in x_], dtype=torch.float)

            x_reco, log_prob_, _ = model(x_, context=params)
            reco_error_ = torch.sum((x_ - x_reco) ** 2, dim=1) ** 0.5

            log_prob.append(log_prob_.detach().numpy())
            reco_error.append(reco_error_.detach().numpy())

        log_prob = np.concatenate(log_prob, axis=0)
        reco_error = np.concatenate(reco_error, axis=0)

    return log_prob, reco_error


def _mcmc(simulator, model=None, n_samples=10, n_mcmc_samples=1000, slice_sampling=False, step=0.2, thin=10, burnin=100):
    # George's settings: thin = 10, n_mcmc_samples = 5000, burnin = 100

    logger.info(
        "Starting MCMC based on %s after %s observed samples, generating %s posterior samples with %s",
        "true simulator likelihood" if model is None else "neural likelihood estimate",
        n_samples,
        n_mcmc_samples,
        "slice sampler" if slice_sampling else "Metropolis-Hastings sampler (step = {})".format(step),
    )

    # Data
    true_parameters = simulator.default_parameters()
    x_obs = load_test_samples(args)[:n_samples]
    x_obs_ = torch.tensor(x_obs, dtype=torch.float)

    if model is None:
        # MCMC based on ground truth likelihood
        def log_posterior(params):
            log_prob = np.sum(simulator.log_density(x_obs, parameters=params))
            log_prob += simulator.evaluate_log_prior(params)
            return float(log_prob)

    else:
        # MCMC based on neural likelihood estimator
        def log_posterior(params):
            params_ = np.broadcast_to(params.reshape((-1, params.shape[-1])), (x_obs.shape[0], params.shape[-1]))
            params_ = torch.tensor(params_, dtype=torch.float)

            if args.algorithm == "flow":
                log_prob = np.sum(model.log_prob(x_obs_, context=params_).detach().numpy())
            elif args.algorithm in ["pie", "slice"]:
                log_prob = np.sum(model.log_prob(x_obs_, context=params_, mode=args.algorithm).detach().numpy())
            else:
                log_prob = np.sum(model.log_prob(x_obs_, context=params_, mode="mf").detach().numpy())

            log_prob += simulator.evaluate_log_prior(params)
            return float(log_prob)

    if slice_sampling:
        logger.debug("Initializing slice sampler")
        sampler = mcmc.SliceSampler(true_parameters, log_posterior, thin=thin)
    else:
        logger.debug("Initializing Gaussian Metropolis-Hastings sampler")
        sampler = mcmc.GaussianMetropolis(true_parameters, log_posterior, step=step, thin=thin)

    if burnin > 0:
        logger.info("Starting burn in")
        sampler.gen(burnin)  # burn in
    logger.info("Burn in done, starting main chain")
    posterior_samples = sampler.gen(n_mcmc_samples)
    logger.info("MCMC done")

    return posterior_samples


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO
    )
    logger.info("Hi!")

    # Model name
    create_modelname(args)
    logger.info("Evaluating model %s", args.modelname)

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Data
    simulator = load_simulator(args)

    # Model
    model = create_model(args, simulator=simulator)
    model.load_state_dict(torch.load(create_filename("model", None, args), map_location=torch.device("cpu")))

    # Evaluate test samples
    log_likelihood_test, reconstruction_error_test = _evaluate_test_samples(args, simulator, model)
    np.save(create_filename("results", "model_log_likelihood_test", args), log_likelihood_test)
    np.save(create_filename("results", "model_reco_error_test", args), reconstruction_error_test)

    try:
        log_likelihood_test, reconstruction_error_test = _evaluate_test_samples(args, simulator, model=None)
        np.save(create_filename("results", "true_log_likelihood_test", args), log_likelihood_test)
        np.save(create_filename("results", "true_reco_error_test", args), reconstruction_error_test)
    except IntractableLikelihoodError:
        logger.info("Ground truth likelihood not tractable, skipping true log likelihood evaluation of test samples")

    # Evaluate either generative or inference performance
    if simulator.parameter_dim() is None:
        # Generate data
        x_gen = _sample_from_model(args, model)

        # Calculate likelihood of data
        _evaluate_model_samples(args, simulator, x_gen)

    else:
        # Evaluate MMD
        if args.skipmodelmcmc:
            logger.info("Skipping MCMC based on model")
            model_posterior_samples = None
        else:
            model_posterior_samples = _mcmc(
                simulator, model, n_samples=args.observedsamples, slice_sampling=args.slicesampler, thin=args.thin, step=args.mcmcstep
            )
            np.save(create_filename("results", "model_posterior_samples", args), model_posterior_samples)

        if args.skiptruthmcmc:
            logger.info("Skipping MCMC based on true likelihood")
        else:
            try:
                true_posterior_samples = _mcmc(simulator, n_samples=args.observedsamples, slice_sampling=args.slicesampler, thin=args.thin, step=args.mcmcstep)
                np.save(create_filename("results", "true_posterior_samples", args), true_posterior_samples)

                if not args.skipmodelmcmc:
                    mmd = sq_maximum_mean_discrepancy(model_posterior_samples, true_posterior_samples, scale="ys")
                    np.save(create_filename("results", "mmd", args), mmd)
                    logger.info("MMD between model and true posterior samples: %s", mmd)

            except IntractableLikelihoodError:
                logger.info("Ground truth likelihood not tractable, skipping MCMC based on true likelihood")

    logger.info("All done! Have a nice day!")
