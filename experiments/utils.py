import numpy as np
import torch
import os
import logging

from experiments.simulators import SphericalGaussianSimulator, ConditionalSphericalGaussianSimulator
from manifold_flow.flows import Flow, PIE, ManifoldFlow
from manifold_flow.training import NumpyDataset

logger = logging.getLogger(__name__)


def _filename(type, label, args):
    if type == "sample":
        filename = "{}/experiments/data/samples/{}/{}_{}_{}_{:.3f}_{}.npy".format(
            args.dir, args.dataset, args.dataset, args.truelatentdim, args.datadim, args.epsilon, label
        )
    elif type == "model":
        filename = "{}/experiments/data/models/{}.pt".format(args.dir, args.modelname)
    elif type == "learning_curve":
        filename = "{}/experiments/data/learning_curves/{}.npy".format(args.dir, args.modelname)
    elif type == "results":
        filename = "{}/experiments/data/results/{}_{}.npy".format(args.dir, args.modelname, label)
    else:
        raise NotImplementedError

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    return filename


def _load_simulator(args):
    if args.dataset == "spherical_gaussian":
        simulator = SphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "conditional_spherical_gaussian":
        simulator = ConditionalSphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    else:
        raise NotImplementedError("Unknown dataset {}".format(args.dataset))
    return simulator


def _create_model(args, context_features):
    if args.algorithm == "flow":
        logger.info("Loading standard flow with %s layers", args.outerlayers)
        model = Flow(data_dim=args.datadim, steps=args.innerlayers + args.outerlayers, transform=args.outertransform, context_features=context_features)
    elif args.algorithm == "pie":
        logger.info("Loading PIE with %s latent dimensions and %s + %s layers", args.modellatentdim, args.outerlayers, args.innerlayers)
        model = PIE(
            data_dim=args.datadim,
            latent_dim=args.modellatentdim,
            steps_inner=args.innerlayers,
            steps_outer=args.outerlayers,
            outer_transform=args.outertransform,
            inner_transform=args.innertransform,
            context_features=context_features,
            apply_context_to_outer=args.conditionalouter,
        )
    elif args.algorithm == "mf":
        logger.info("Loading manifold flow with %s latent dimensions and %s + %s layers", args.modellatentdim, args.outerlayers, args.innerlayers)
        model = ManifoldFlow(
            data_dim=args.datadim,
            latent_dim=args.modellatentdim,
            steps_inner=args.innerlayers,
            steps_outer=args.outerlayers,
            outer_transform=args.outertransform,
            inner_transform=args.innertransform,
            context_features=context_features,
            apply_context_to_outer=args.conditionalouter,
        )
    else:
        raise NotImplementedError("Unknown algorithm {}".format(args.algorithm))
    return model


def _load_training_dataset(args):
    if args.dataset == "spherical_gaussian":
        x = np.load(_filename("sample", "x_train", args))
        params = np.ones(x.shape[0])
        dataset = NumpyDataset(x, params)
    elif args.dataset == "conditional_spherical_gaussian":
        x = np.load(_filename("sample", "x_train", args))
        params = np.load(_filename("sample", "parameters_train", args))
        dataset = NumpyDataset(x, params)
    else:
        raise NotImplementedError("Unknown dataset {}".format(args.dataset))
    return dataset


def _load_test_samples(args):
    return np.load(_filename("sample", "x_test", args))


def _create_modelname(args):
    if args.modelname is None:
        args.modelname = "{}_{}_{}_{}_{}_{:.3f}".format(args.algorithm, args.modellatentdim, args.dataset, args.truelatentdim, args.datadim, args.epsilon)
    logger.info(
        "Evaluating inference for model %s on data set %s (data dim %s, true latent dim %s)", args.modelname, args.dataset, args.datadim, args.truelatentdim
    )
