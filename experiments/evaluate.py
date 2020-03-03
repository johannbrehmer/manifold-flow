#! /usr/bin/env python

import numpy as np
import logging
import sys
import torch
import argparse
import copy

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
    parser.add_argument("--truth", action="store_true")
    parser.add_argument("--modelname", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default="mf", choices=ALGORITHMS)
    parser.add_argument("--dataset", type=str, default="spherical_gaussian", choices=SIMULATORS)
    parser.add_argument("-i", type=int, default=0)

    # Dataset details
    parser.add_argument("--truelatentdim", type=int, default=2)
    parser.add_argument("--datadim", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.01)

    # Model details
    parser.add_argument("--modellatentdim", type=int, default=2)
    parser.add_argument("--specified", action="store_true")
    parser.add_argument("--outertransform", type=str, default="rq-coupling")
    parser.add_argument("--innertransform", type=str, default="rq-coupling")
    parser.add_argument("--lineartransform", type=str, default="permutation")
    parser.add_argument("--outerlayers", type=int, default=5)
    parser.add_argument("--innerlayers", type=int, default=5)
    parser.add_argument("--conditionalouter", action="store_true")
    parser.add_argument("--outercouplingmlp", action="store_true")
    parser.add_argument("--outercouplinglayers", type=int, default=2)
    parser.add_argument("--outercouplinghidden", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pieepsilon", type=float, default=0.01)
    parser.add_argument("--encoderblocks", type=int, default=5)
    parser.add_argument("--encoderhidden", type=int, default=100)
    parser.add_argument("--encodermlp", action="store_true")
    parser.add_argument("--splinerange", default=3.0, type=float)
    parser.add_argument("--splinebins", default=8, type=int)

    # Evaluation settings
    parser.add_argument("--gridresolution", type=int, default=11)
    parser.add_argument("--generate", type=int, default=10000)
    parser.add_argument("--observedsamples", type=int, default=20)
    parser.add_argument("--slicesampler", action="store_true")
    parser.add_argument("--mcmcstep", type=float, default=0.15)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--mcmcsamples", type=int, default=5000)
    parser.add_argument("--burnin", type=int, default=100)

    # Other settings
    parser.add_argument("--dir", type=str, default="/scratch/jb6504/manifold-flow")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skipmodelmcmc", action="store_true")
    parser.add_argument("--skiptruthmcmc", action="store_true")

    return parser.parse_args()


def _sample_from_model(args, model, simulator):
    logger.info("Sampling from model")
    if simulator.parameter_dim() is None:
        x_gen = model.sample(n=args.generate).detach().numpy()
    else:
        params = simulator.default_parameters()
        params = np.asarray([params for _ in range(args.generate)])
        params = torch.tensor(params, dtype=torch.float)
        x_gen = model.sample(n=args.generate, context=params).detach().numpy()
    np.save(create_filename("results", "samples", args), x_gen)
    return x_gen


def _evaluate_model_samples(args, simulator, x_gen):
    # Likelihood
    logger.info("Calculating likelihood of generated samples")
    if simulator.parameter_dim() is None:
        log_likelihood_gen = simulator.log_density(x_gen)
    else:
        params = simulator.default_parameters()
        params = np.asarray([params for _ in range(args.generate)])
        log_likelihood_gen = simulator.log_density(x_gen, parameters=params)
    log_likelihood_gen[np.isnan(log_likelihood_gen)] = -1.0e-12
    np.save(create_filename("results", "samples_likelihood", args), log_likelihood_gen)

    # Distance from manifold
    try:
        logger.info("Calculating distance from manifold of generated samples")
        distances_gen = simulator.distance_from_manifold(x_gen)
        np.save(create_filename("results", "samples_manifold_distance", args), distances_gen)
    except NotImplementedError:
        logger.info("Cannot calculate distance from manifold for dataset %s", args.dataset)


def _evaluate_test_samples(args, simulator, model=None, samples=1000, batchsize=100, ood=False):
    if model is None:
        logger.info("Evaluating true log likelihood of test samples")
    else:
        logger.info("Evaluating model likelihood of test samples")

    x = load_test_samples(simulator, args, ood=ood)[:samples]
    parameter_grid = [None] if simulator.parameter_dim() is None else simulator.eval_parameter_grid(resolution=args.gridresolution)

    log_probs = []
    reco_error = None

    for i, params in enumerate(parameter_grid):
        logger.debug("Evaluating grid point %s / %s", i + 1, len(parameter_grid))
        if model is None:
            params_ = None if params is None else np.asarray([params for _ in x])
            log_prob = simulator.log_density(x, parameters=params_)
            if reco_error is None:
                reco_error = np.zeros(x.shape[0])

        else:
            log_prob = []
            reco_error_ = []
            n_batches = (samples - 1) // batchsize + 1

            for j in range(n_batches):
                x_ = torch.tensor(x[j * batchsize : (j + 1) * batchsize], dtype=torch.float)
                if params is None:
                    params_ = None
                else:
                    params_ = np.asarray([params for _ in x_])
                    params_ = torch.tensor(params_, dtype=torch.float)

                if args.algorithm == "flow":
                    x_reco, log_prob_, _ = model(x_, context=params_)
                elif args.algorithm in ["pie", "slice"]:
                    x_reco, log_prob_, _ = model(x_, context=params_, mode=args.algorithm)
                else:
                    x_reco, log_prob_, _ = model(x_, context=params_, mode="mf")

                log_prob.append(log_prob_.detach().numpy())
                reco_error_.append((torch.sum((x_ - x_reco) ** 2, dim=1) ** 0.5).detach().numpy())

            log_prob = np.concatenate(log_prob, axis=0)
            if reco_error is None:
                reco_error = np.concatenate(reco_error_, axis=0)

        log_probs.append(log_prob)

    if simulator.parameter_dim() is None:
        return np.asarray(log_probs[0]), reco_error, None
    return np.asarray(log_probs), reco_error, parameter_grid


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
    x_obs = load_test_samples(simulator, args)[:n_samples]
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
    logger.info("Starting evaluate.py with arguments %s", args)

    if args.truth:
        create_modelname(args)
        logger.info("Evaluating simulator truth")
    else:
        create_modelname(args)
        logger.info("Evaluating model %s", args.modelname)

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Data
    simulator = load_simulator(args)

    # Load model
    if not args.truth:
        model = create_model(args, simulator=simulator)
        model.load_state_dict(torch.load(create_filename("model", None, args), map_location=torch.device("cpu")))
        model.eval()
    else:
        model = None

    if args.truth:
        # Evaluate test samples with true likelihood
        try:
            log_likelihood_test, reconstruction_error_test, parameter_grid = _evaluate_test_samples(args, simulator, model=None)
            np.save(create_filename("results", "true_log_likelihood_test", args), log_likelihood_test)
        except IntractableLikelihoodError:
            logger.info("Ground truth likelihood not tractable, skipping true log likelihood evaluation of test samples")

    else:
        # Evaluate test samples
        log_likelihood_test, reconstruction_error_test, parameter_grid = _evaluate_test_samples(args, simulator, model)
        np.save(create_filename("results", "model_log_likelihood_test", args), log_likelihood_test)
        np.save(create_filename("results", "model_reco_error_test", args), reconstruction_error_test)
        if parameter_grid is not None:
            np.save(create_filename("results", "parameter_grid_test", args), parameter_grid)

        # Evaluate ood samples
        try:
            log_likelihood_ood, reconstruction_error_ood, _ = _evaluate_test_samples(args, simulator, model, ood=True)
            np.save(create_filename("results", "model_log_likelihood_ood", args), log_likelihood_ood)
            np.save(create_filename("results", "model_reco_error_ood", args), reconstruction_error_ood)
        except:
            pass

    # Evaluate generative performance
    if not args.truth:
        x_gen = _sample_from_model(args, model, simulator)
        _evaluate_model_samples(args, simulator, x_gen)

    # Truth MCMC
    if simulator.parameter_dim() is not None and args.truth:
        # Truth MCMC
        try:
            true_posterior_samples = _mcmc(
                simulator,
                n_mcmc_samples=args.mcmcsamples,
                n_samples=args.observedsamples,
                slice_sampling=args.slicesampler,
                thin=args.thin,
                step=args.mcmcstep,
                burnin=args.burnin,
            )
            np.save(create_filename("results", "posterior_samples", args), true_posterior_samples)

        except IntractableLikelihoodError:
            logger.info("Ground truth likelihood not tractable, skipping MCMC based on true likelihood")

    # Evaluate inference performance
    if simulator.parameter_dim() is not None and not args.truth:
        # MCMC
        model_posterior_samples = _mcmc(
            simulator,
            model,
            n_mcmc_samples=args.mcmcsamples,
            n_samples=args.observedsamples,
            slice_sampling=args.slicesampler,
            thin=args.thin,
            step=args.mcmcstep,
            burnin=args.burnin,
        )
        np.save(create_filename("results", "posterior_samples", args), model_posterior_samples)

        # MMD calculation
        args_ = copy.deepcopy(args)
        args_.truth = True
        args_.modelname = None
        create_modelname(args_)
        true_posterior_samples = np.load(create_filename("results", "posterior_samples", args_))

        mmd = sq_maximum_mean_discrepancy(model_posterior_samples, true_posterior_samples, scale="ys")
        np.save(create_filename("results", "mmd", args), mmd)
        logger.info("MMD between model and true posterior samples: %s", mmd)

    logger.info("All done! Have a nice day!")
