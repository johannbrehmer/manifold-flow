import numpy as np
import logging
import copy

from experiments.simulators import (
    SphericalGaussianSimulator,
    ConditionalSphericalGaussianSimulator,
    CIFAR10Loader,
    ImageNetLoader,
    WBFLoader,
    WBF2DLoader,
    PowerManifoldSimulator,
)
from experiments.utils import SIMULATORS
from experiments.utils.names import create_filename
from manifold_flow.training import NumpyDataset

logger = logging.getLogger(__name__)


def load_simulator(args):
    assert args.dataset in SIMULATORS
    if args.dataset == "power":
        simulator = PowerManifoldSimulator(filename=args.dir + "/experiments/data/samples/power/manifold.npz")
    elif args.dataset == "spherical_gaussian":
        simulator = SphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "conditional_spherical_gaussian":
        simulator = ConditionalSphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "lhc":
        simulator = WBFLoader()
    elif args.dataset == "lhc2d":
        simulator = WBF2DLoader()
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


def load_test_samples(simulator, args, ood=False):
    try:
        return simulator.load_dataset(train=False, dataset_dir=create_filename("dataset", None, args_))
    except NotImplementedError:
        # We want to always use the i=0 test samples for a better comparison
        args_ = copy.deepcopy(args)
        args_.i = 0

        return np.load(create_filename("sample", "x_ood" if ood else "x_test", args_))
