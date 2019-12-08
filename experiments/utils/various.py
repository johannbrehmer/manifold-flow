import numpy as np
import os
import logging

from experiments.simulators import SphericalGaussianSimulator, ConditionalSphericalGaussianSimulator, CIFAR10Loader, ImageNetLoader
from manifold_flow.training import NumpyDataset

logger = logging.getLogger(__name__)


SIMULATORS = ["spherical_gaussian", "conditional_spherical_gaussian", "cifar10", "imagenet"]
ALGORITHMS = ["flow", "pie", "mf", "slice", "gamf", "hybrid"]


def create_filename(type, label, args):
    if type == "dataset":
        filename = "{}/experiments/data/samples/{}".format(args.dir, args.dataset)
    elif type == "sample":
        filename = "{}/experiments/data/samples/{}/{}_{}_{}_{:.3f}_{}.npy".format(
            args.dir, args.dataset, args.dataset, args.truelatentdim, args.datadim, args.epsilon, label
        )
    elif type == "model":
        filename = "{}/experiments/data/models/{}.pt".format(args.dir, args.modelname)
    elif type == "learning_curve":
        filename = "{}/experiments/data/learning_curves/{}.npy".format(args.dir, args.modelname)
    elif type == "results":
        filename = "{}/experiments/data/results/{}_{}.npy".format(args.dir, args.modelname, label)
    elif type == "timing":
        filename = "{}/experiments/data/timing/{}_{}_{}_{}_{}_{}.npy".format(
            args.dir,
            args.algorithm,
            args.outerlayers,
            args.outertransform,
            "mlp" if args.outercouplingmlp else "resnet",
            args.outercouplinglayers,
            args.outercouplinghidden,
        )
    else:
        raise NotImplementedError

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    return filename


def create_modelname(args):
    if args.modelname is None:
        args.modelname = "{}_{}_{}_{}_{}_{:.3f}".format(args.algorithm, args.modellatentdim, args.dataset, args.truelatentdim, args.datadim, args.epsilon)


def load_simulator(args):
    assert args.dataset in SIMULATORS
    if args.dataset == "spherical_gaussian":
        simulator = SphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "conditional_spherical_gaussian":
        simulator = ConditionalSphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "cifar10":
        simulator = CIFAR10Loader()
        args.datadim = simulator.data_dim()
    elif args.dataset == "imagenet":
        simulator = ImageNetLoader()
        args.datadim = simulator.data_dim()
    else:
        raise NotImplementedError("Unknown dataset {}".format(args.dataset))
    return simulator


def load_training_dataset(simulator, args):
    try:
        return simulator.load_dataset(train=True, dataset_dir=create_filename("dataset", None, args))
    except:
        pass

    if args.dataset == "spherical_gaussian":
        x = np.load(create_filename("sample", "x_train", args))
    try:
        params = np.load(create_filename("sample", "parameters_train", args))
    except:
        params = np.ones(x.shape[0])

    return NumpyDataset(x, params)


def load_test_samples(args):
    return np.load(create_filename("sample", "x_test", args))
