#! /usr/bin/env python

""" Top-level script for evaluating models """

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
    """ Parses command line arguments for the evaluation """

    parser = argparse.ArgumentParser()

    # What what what
    parser.add_argument("--truth", action="store_true", help="Evaluate ground truth rather than learned model")
    parser.add_argument("--modelname", type=str, default=None, help="Model name. Algorithm, latent dimension, dataset, and run are prefixed automatically.")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="flow",
        choices=ALGORITHMS,
        help="Algorithm: flow (for AF), mf (for FOM, MFMF), emf (for MFMFE), pie (for PIE), gamf (for MFMF-OT)...",
    )
    parser.add_argument(
        "--dataset", type=str, default="spherical_gaussian", choices=SIMULATORS, help="Dataset: spherical_gaussian, power, lhc, lhc40d, lhc2d, and some others"
    )
    parser.add_argument("-i", type=int, default=0, help="Run number")

    # Dataset details
    parser.add_argument("--truelatentdim", type=int, default=2, help="True manifold dimensionality (for datasets where that is variable)")
    parser.add_argument("--datadim", type=int, default=3, help="True data dimensionality (for datasets where that is variable)")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Noise term (for datasets where that is variable)")

    # Model details
    parser.add_argument("--modellatentdim", type=int, default=2, help="Model manifold dimensionality")
    parser.add_argument("--specified", action="store_true", help="Prescribe manifold chart: FOM instead of MFMF")
    parser.add_argument(
        "--outertransform",
        type=str,
        default="rq-coupling",
        help="Type of transformations for f: affine-coupling, quadratic-coupling, rq-coupling, affine-autoregressive, quadratic-autoregressive, rq-autoregressive. See Neural Spline Flow paper.",
    )
    parser.add_argument(
        "--innertransform",
        type=str,
        default="rq-coupling",
        help="Type of transformations for h: affine-coupling, quadratic-coupling, rq-coupling, affine-autoregressive, quadratic-autoregressive, rq-autoregressive. See Neural Spline Flow paper.",
    )
    parser.add_argument(
        "--lineartransform",
        type=str,
        default="permutation",
        help="Type of linear transformations inserted between the base transformations: linear, permutation. See Neural Spline Flow paper.",
    )
    parser.add_argument("--outerlayers", type=int, default=5, help="Number of transformations in f (not counting linear transformations)")
    parser.add_argument("--innerlayers", type=int, default=5, help="Number of transformations in h (not counting linear transformations)")
    parser.add_argument(
        "--conditionalouter", action="store_true", help="If dataset is conditional, use this to make f conditional (otherwise only h is conditional)"
    )
    parser.add_argument("--outercouplingmlp", action="store_true", help="Use MLP instead of ResNet for coupling layers")
    parser.add_argument("--outercouplinglayers", type=int, default=2, help="Number of layers for coupling layers")
    parser.add_argument("--outercouplinghidden", type=int, default=100, help="Number of hidden units for coupling layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Use dropout")
    parser.add_argument("--pieepsilon", type=float, default=0.01, help="PIE epsilon term")
    parser.add_argument("--encoderblocks", type=int, default=5, help="Number of blocks in MFMFE encoder")
    parser.add_argument("--encoderhidden", type=int, default=100, help="Number of hidden units in MFMFE encoder")
    parser.add_argument("--encodermlp", action="store_true", help="Use MLP instead of ResNet for MFMFE encoder")
    parser.add_argument("--splinerange", default=3.0, type=float, help="Spline boundaries")
    parser.add_argument("--splinebins", default=8, type=int, help="Number of spline bins")

    # Evaluation settings
    parser.add_argument("--gridresolution", type=int, default=11, help="Grid ressolution (per axis) for likelihood eval")
    parser.add_argument("--generate", type=int, default=10000, help="Number of samples for generative mode eval")
    parser.add_argument("--observedsamples", type=int, default=20, help="Number of iid samples in synthetic 'observed' set for inference tasks")
    parser.add_argument("--slicesampler", action="store_true", help="Use slice sampler for MCMC")
    parser.add_argument("--mcmcstep", type=float, default=0.15, help="MCMC step size")
    parser.add_argument("--thin", type=int, default=1, help="MCMC thinning")
    parser.add_argument("--mcmcsamples", type=int, default=5000, help="Length of MCMC chain")
    parser.add_argument("--burnin", type=int, default=100, help="MCMC burn in")
    parser.add_argument("--evalbatchsize", type=int, default=100, help="Likelihood eval batch size")
    parser.add_argument("--chain", type=int, default=0, help="MCMC chain")
    parser.add_argument("--trueparam", type=int, default=0, help="Index of true parameter point for inference tasks")

    # Other settings
    parser.add_argument("--dir", type=str, default="/scratch/jb6504/manifold-flow", help="Base directory of repo")
    parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")
    parser.add_argument("--skipgeneration", action="store_true", help="Skip generative mode eval")
    parser.add_argument("--skiplikelihood", action="store_true", help="Skip likelihood eval")
    parser.add_argument("--skipood", action="store_true", help="Skip OOD likelihood eval")
    parser.add_argument("--skipinference", action="store_true", help="Skip all inference tasks (likelihood eval and MCMC)")
    parser.add_argument("--skipmcmc", action="store_true", help="Skip MCMC")

    return parser.parse_args()


def sample_from_model(args, model, simulator):
    """ Generate samples from model and store """

    logger.info("Sampling from model")
    if simulator.parameter_dim() is None:
        x_gen = model.sample(n=args.generate).detach().numpy()
    else:
        params = simulator.default_parameters(true_param_id=args.trueparam)
        params = np.asarray([params for _ in range(args.generate)])
        params = torch.tensor(params, dtype=torch.float)
        x_gen = model.sample(n=args.generate, context=params).detach().numpy()
    np.save(create_filename("results", "samples", args), x_gen)
    return x_gen


def evaluate_model_samples(args, simulator, x_gen):
    """ Evaluate model samples and save results """

    logger.info("Calculating likelihood of generated samples")

    try:
        if simulator.parameter_dim() is None:
            log_likelihood_gen = simulator.log_density(x_gen)
        else:
            params = simulator.default_parameters(true_param_id=args.trueparam)
            params = np.asarray([params for _ in range(args.generate)])
            log_likelihood_gen = simulator.log_density(x_gen, parameters=params)
        log_likelihood_gen[np.isnan(log_likelihood_gen)] = -1.0e-12
        np.save(create_filename("results", "samples_likelihood", args), log_likelihood_gen)
    except IntractableLikelihoodError:
        logger.info("True simulator likelihood is intractable for dataset %s", args.dataset)

    # Distance from manifold
    try:
        logger.info("Calculating distance from manifold of generated samples")
        distances_gen = simulator.distance_from_manifold(x_gen)
        np.save(create_filename("results", "samples_manifold_distance", args), distances_gen)
    except NotImplementedError:
        logger.info("Cannot calculate distance from manifold for dataset %s", args.dataset)


def evaluate_test_samples(args, simulator, model=None, samples=1000, batchsize=100, ood=False, paramscan=False):
    """ Likelihood evaluation """

    logger.info("Evaluating %s likelihood of %s samples", "true" if model is None else "model", "ood" if ood else "test")

    x = load_test_samples(simulator, args, ood=ood, paramscan=paramscan)[:samples]
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


def run_mcmc(args, simulator, model=None):
    """ MCMC """

    logger.info(
        "Starting MCMC based on %s after %s observed samples, generating %s posterior samples with %s for parameter point number %s",
        "true simulator likelihood" if model is None else "neural likelihood estimate",
        args.observedsamples,
        args.mcmcsamples,
        "slice sampler" if args.slicesampler else "Metropolis-Hastings sampler (step = {})".format(args.mcmcstep),
        args.trueparam,
    )

    # Data
    true_parameters = simulator.default_parameters(true_param_id=args.trueparam)
    x_obs = load_test_samples(simulator, args)[: args.observedsamples]
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

    if args.slicesampler:
        logger.debug("Initializing slice sampler")
        sampler = mcmc.SliceSampler(true_parameters, log_posterior, thin=args.thin)
    else:
        logger.debug("Initializing Gaussian Metropolis-Hastings sampler")
        sampler = mcmc.GaussianMetropolis(true_parameters, log_posterior, step=args.mcmcstep, thin=args.thin)

    if args.burnin > 0:
        logger.info("Starting burn in")
        sampler.gen(args.burnin)
    logger.info("Burn in done, starting main chain")
    posterior_samples = sampler.gen(args.mcmcsamples)
    logger.info("MCMC done")

    return posterior_samples


if __name__ == "__main__":
    # Parse args
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO
    )
    logger.info("Hi!")
    logger.debug("Starting evaluate.py with arguments %s", args)

    # Model name
    if args.truth:
        create_modelname(args)
        logger.info("Evaluating simulator truth")
    else:
        create_modelname(args)
        logger.info("Evaluating model %s", args.modelname)

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Data set
    simulator = load_simulator(args)

    # Load model
    if not args.truth:
        model = create_model(args, simulator=simulator)
        model.load_state_dict(torch.load(create_filename("model", None, args), map_location=torch.device("cpu")))
        model.eval()
    else:
        model = None

    # Evaluate generative performance
    if args.skipgeneration:
        logger.info("Skipping generative evaluation as per request.")
    elif not args.truth:
        x_gen = sample_from_model(args, model, simulator)
        evaluate_model_samples(args, simulator, x_gen)

    if args.skipinference:
        logger.info("Skipping all inference tasks as per request. Have a nice day!")
        exit()

    # Evaluate test and ood samples
    if args.skiplikelihood:
        logger.info("Skipping likelihood evaluation on test and OOD samples as per request")

    elif args.truth:
        try:
            log_likelihood_test, reconstruction_error_test, parameter_grid = evaluate_test_samples(args, simulator, model=None, batchsize=args.evalbatchsize)
            np.save(create_filename("results", "true_log_likelihood_test", args), log_likelihood_test)

            if args.skipood:
                logger.info("Skipping OOD evaluation")
            else:
                log_likelihood_ood, _, _ = evaluate_test_samples(args, simulator, model=None, batchsize=args.evalbatchsize)
                np.save(create_filename("results", "true_log_likelihood_ood", args), log_likelihood_ood)
        except IntractableLikelihoodError:
            logger.info("Ground truth likelihood not tractable, skipping true log likelihood evaluation of test samples")

    else:
        log_likelihood_test, reconstruction_error_test, parameter_grid = evaluate_test_samples(args, simulator, model, batchsize=args.evalbatchsize)
        np.save(create_filename("results", "model_log_likelihood_test", args), log_likelihood_test)
        np.save(create_filename("results", "model_reco_error_test", args), reconstruction_error_test)
        if parameter_grid is not None:
            np.save(create_filename("results", "parameter_grid_test", args), parameter_grid)

        if args.skipood:
            logger.info("Skipping OOD evaluation")
        else:
            try:
                log_likelihood_ood, reconstruction_error_ood, _ = evaluate_test_samples(args, simulator, model, ood=True, batchsize=args.evalbatchsize)
                np.save(create_filename("results", "model_log_likelihood_ood", args), log_likelihood_ood)
                np.save(create_filename("results", "model_reco_error_ood", args), reconstruction_error_ood)
            except:
                pass

    if args.skipmcmc:
        logger.info("Skipping MCMC as per request")

    # Truth MCMC
    elif simulator.parameter_dim() is not None and args.truth:
        try:
            true_posterior_samples = run_mcmc(args, simulator)
            np.save(create_filename("mcmcresults", "posterior_samples", args), true_posterior_samples)

        except IntractableLikelihoodError:
            logger.info("Ground truth likelihood not tractable, skipping MCMC based on true likelihood")

    # Model-based MCMC
    elif simulator.parameter_dim() is not None and not args.truth:
        model_posterior_samples = run_mcmc(args, simulator, model)
        np.save(create_filename("mcmcresults", "posterior_samples", args), model_posterior_samples)

        # MMD calculation (only accurate if there is only one chain)
        args_ = copy.deepcopy(args)
        args_.truth = True
        args_.modelname = None
        create_modelname(args_)
        try:
            true_posterior_samples = np.load(create_filename("mcmcresults", "posterior_samples", args_))

            mmd = sq_maximum_mean_discrepancy(model_posterior_samples, true_posterior_samples, scale="ys")
            np.save(create_filename("results", "mmd", args), mmd)
            logger.info("MMD between model and true posterior samples: %s", mmd)
        except FileNotFoundError:
            logger.info("No true posterior data, skipping MMD calculation!")

    logger.info("All done! Have a nice day!")
