import copy
import numpy as np
import logging

from .base import IntractableLikelihoodError
from .spherical_simulator import SphericalGaussianSimulator
from .conditional_spherical_simulator import ConditionalSphericalGaussianSimulator
from .images import ImageNetLoader, CelebALoader, FFHQStyleGAN2DLoader, IMDBLoader
from .collider import WBFLoader, WBF2DLoader, WBF40DLoader
from .polynomial_surface_simulator import PolynomialSurfaceSimulator
from .lorenz import LorenzSimulator
from experiments.utils import create_filename
from .utils import NumpyDataset

logger = logging.getLogger(__name__)


SIMULATORS = ["power", "spherical_gaussian", "conditional_spherical_gaussian", "lhc", "lhc40d", "lhc2d", "imagenet", "celeba", "gan2d", "lorenz", "imdb"]


def load_simulator(args):
    assert args.dataset in SIMULATORS
    if args.dataset == "power":
        simulator = PolynomialSurfaceSimulator(filename=args.dir + "/experiments/data/samples/power/manifold.npz")
    elif args.dataset == "spherical_gaussian":
        simulator = SphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "conditional_spherical_gaussian":
        simulator = ConditionalSphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "lhc":
        simulator = WBFLoader()
    elif args.dataset == "lhc2d":
        simulator = WBF2DLoader()
    elif args.dataset == "lhc40d":
        simulator = WBF40DLoader()
    elif args.dataset == "imagenet":
        simulator = ImageNetLoader()
    elif args.dataset == "celeba":
        simulator = CelebALoader()
    elif args.dataset == "gan2d":
        simulator = FFHQStyleGAN2DLoader()
    elif args.dataset == "lorenz":
        simulator = LorenzSimulator()
    elif args.dataset == "imdb":
        simulator = IMDBLoader()
    else:
        raise ValueError("Unknown dataset {}".format(args.dataset))

    args.datadim = simulator.data_dim()
    return simulator


def load_training_dataset(simulator, args):
    try:
        return simulator.load_dataset(
            train=True, dataset_dir=create_filename("dataset", None, args), limit_samplesize=args.samplesize, joint_score=args.scandal is not None
        )
    except NotImplementedError:
        pass

    x = np.load(create_filename("sample", "x_train", args))
    try:
        params = np.load(create_filename("sample", "parameters_train", args))
    except:
        params = np.ones(x.shape[0])
    if args.scandal is not None:
        raise NotImplementedError("SCANDAL training not implemented for this dataset")

    if args.samplesize is not None:
        logger.info("Only using %s of %s available samples", args.samplesize, x.shape[0])
        x = x[: args.samplesize]
        params = params[: args.samplesize]

    return NumpyDataset(x, params)


def load_test_samples(simulator, args, ood=False, paramscan=False, limit_samplesize=None):
    try:
        x, _ = simulator.load_dataset(
            train=False,
            numpy=True,
            dataset_dir=create_filename("dataset", None, args),
            true_param_id=args.trueparam,
            joint_score=False,
            limit_samplesize=limit_samplesize,
        )

        # TODO: implement OOD
        return x

    except NotImplementedError:
        # We want to always use the i=0 test samples for a better comparison
        args_ = copy.deepcopy(args)
        args_.i = 0

        x = np.load(create_filename("sample", "x_ood" if ood else "x_paramscan" if paramscan else "x_test", args_))
        if limit_samplesize is None:
            x = x[:limit_samplesize]
        return x
