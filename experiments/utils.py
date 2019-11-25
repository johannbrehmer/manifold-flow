import numpy as np
import torch
import os

from experiments.evaluate import logger
from experiments.simulators import SphericalGaussianSimulator
from experiments.train import logger
from manifold_flow.flows import Flow, PIE, ManifoldFlow
from manifold_flow.training import NumpyDataset


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
    else:
        raise NotImplementedError("Unknown dataset {}".format(args.dataset))
    return simulator


def _load_model(args):
    if args.algorithm == "flow":
        logger.info("Loading standard flow with %s layers", args.outerlayers)
        model = Flow(data_dim=args.datadim, steps=args.outerlayers, transform=args.transform)
    elif args.algorithm == "pie":
        logger.info("Loading PIE with %s latent dimensions and %s + %s layers", args.modellatentdim, args.outerlayers,
                    args.innerlayers)
        model = PIE(
            data_dim=args.datadim,
            latent_dim=args.modellatentdim,
            steps_inner=args.innerlayers,
            steps_outer=args.outerlayers,
            outer_transform=args.transform,
            inner_transform=args.transform,
        )
    elif args.algorithm == "mf":
        logger.info("Loading manifold flow with %s latent dimensions and %s + %s layers", args.modellatentdim,
                    args.outerlayers, args.innerlayers)
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
    model.load_state_dict(torch.load("{}/experiments/data/models/{}.pt".format(args.dir, args.modelname),
                                     map_location=torch.device("cpu")))
    return model


def _load_training_data(args):
    if args.dataset == "spherical_gaussian":
        x = np.load(
            "{}/experiments/data/samples/spherical_gaussian/spherical_gaussian_{}_{}_{:.3f}_x_train.npy".format(
                args.dir, args.truelatentdim, args.datadim, args.epsilon
            )
        )
        y = np.ones(x.shape[0])
        dataset = NumpyDataset(x, y)
        logger.info("Loaded spherical Gaussian data")
    else:
        raise NotImplementedError("Unknown dataset {}".format(args.dataset))
    return dataset
