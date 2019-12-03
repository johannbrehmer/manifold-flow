import numpy as np
import os
import logging

from experiments.simulators import SphericalGaussianSimulator, ConditionalSphericalGaussianSimulator
from experiments.utils import vector_transforms
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
        logger.info("Creating standard flow with %s layers, transform %s, %s context features", args.innerlayers + args.outerlayers, args.outertransform, context_features)
        transform = vector_transforms.create_transform(args.datadim, args.innerlayers + args.outerlayers, base_transform_type=args.outertransform, context_features=context_features)
        model = Flow(data_dim=args.datadim, transform=transform)

    elif args.algorithm == "pie":
        logger.info("Creating PIE with %s latent dimensions, %s + %s layers, transforms %s / %s, %s context features", args.modellatentdim, args.outerlayers, args.innerlayers, args.outertransform, args.innertransform, context_features)

        outer_transform = vector_transforms.create_transform(
            args.datadim, args.outerlayers, base_transform_type=args.outertransform, context_features=context_features if args.conditionalouter else None
        )
        inner_transform = vector_transforms.create_transform(args.modellatentdim, args.innerlayers, base_transform_type=args.innertransform, context_features=context_features)
        model = PIE(data_dim = args.datadim, latent_dim = args.modellatentdim, outer_transform=outer_transform, inner_transform=inner_transform, apply_context_to_outer=args.conditionalouter)

    elif args.algorithm == "mf":
        logger.info("Creating manifold flow with %s latent dimensions, %s + %s layers, transforms %s / %s, %s context features", args.modellatentdim, args.outerlayers, args.innerlayers, args.outertransform, args.innertransform, context_features)
        outer_transform_kwargs = {}
        try:
            outer_transform_kwargs["hidden_features"] = args.outercouplinghidden
            outer_transform_kwargs["num_transform_blocks"] = args.outercouplinglayers
            outer_transform_kwargs["resnet_transform"] = not args.outercouplingmlp
            logger.info("Additional settings for outer transform: %s", outer_transform_kwargs)
        except:
            pass
        outer_transform = vector_transforms.create_transform(
            args.datadim, args.outerlayers, base_transform_type=args.outertransform, context_features=context_features if args.conditionalouter else None, **outer_transform_kwargs
        )
        inner_transform = vector_transforms.create_transform(args.modellatentdim, args.innerlayers, base_transform_type=args.innertransform, context_features=context_features)
        model = ManifoldFlow(
            data_dim=args.datadim,
            latent_dim=args.modellatentdim,
            outer_transform=outer_transform,
            inner_transform=inner_transform,
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
