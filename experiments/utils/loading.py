import numpy as np
import logging

from experiments.simulators import SphericalGaussianSimulator, ConditionalSphericalGaussianSimulator, CIFAR10Loader, ImageNetLoader
from experiments.utils import SIMULATORS
from experiments.utils.names import create_filename
from manifold_flow.training import NumpyDataset

logger = logging.getLogger(__name__)


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
