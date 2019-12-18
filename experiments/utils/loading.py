import numpy as np
import logging

from experiments.simulators import SphericalGaussianSimulator, ConditionalSphericalGaussianSimulator, CIFAR10Loader, ImageNetLoader, TopHiggsLoader
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
    elif args.dataset == "tth":
        simulator = TopHiggsLoader()
    elif args.dataset == "cifar10":
        simulator = CIFAR10Loader()
    elif args.dataset == "imagenet":
        simulator = ImageNetLoader()
    else:
        raise ValueError("Unknown dataset {}".format(args.dataset))

    args.datadim = simulator.data_dim()
    return simulator


def load_training_dataset(simulator, args):
    try:
        return simulator.load_dataset(train=True, dataset_dir=create_filename("dataset", None, args), limit_samplesize=args.samplesize)
    except NotImplementedError:
        pass

    if args.dataset == "spherical_gaussian":
        x = np.load(create_filename("sample", "x_train", args))
    try:
        params = np.load(create_filename("sample", "parameters_train", args))
    except:
        params = np.ones(x.shape[0])

    if args.samplesize is not None:
        logger.info("Only using %s of %s available samples", args.samplesize, x.shape[0])
        x = x[: args.samplesize]
        params = params[: args.samplesize]

    return NumpyDataset(x, params)


def load_test_samples(args):
    return np.load(create_filename("sample", "x_test", args))
