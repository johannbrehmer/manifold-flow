import os
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def create_filename(type_, label, args):
    run_label = "_run{}".format(args.i) if args.i > 0 else ""

    if type_ == "dataset":  # Fixed datasets
        filename = "{}/experiments/data/samples/{}".format(args.dir, args.dataset)

    elif type_ == "sample":  # Dynamically sampled from simulator
        if args.dataset in ["spherical_gaussian", "conditional_spherical_gaussian"]:
            filename = "{}/experiments/data/samples/{}/{}_{}_{}_{:.3f}_{}{}.npy".format(
                args.dir, args.dataset, args.dataset, args.truelatentdim, args.datadim, args.epsilon, label, run_label
            )
        else:
            filename = "{}/experiments/data/samples/{}/{}{}.npy".format(args.dir, args.dataset, label, run_label)

    elif type_ == "model":
        filename = "{}/experiments/data/models/{}.pt".format(args.dir, args.modelname)

    elif type_ == "checkpoint":
        filename = "{}/experiments/data/models/checkpoints/{}_{}_{}.pt".format(args.dir, args.modelname, "epoch" if label is None else "epoch_" + label, "{}")

    elif type_ == "training_plot":
        filename = "{}/experiments/figures/training/{}_{}_{}.pdf".format(args.dir, args.modelname, "epoch" if label is None else label, "{}")

    elif type_ == "learning_curve":
        filename = "{}/experiments/data/learning_curves/{}.npy".format(args.dir, args.modelname)

    elif type_ == "results":
        trueparam_name = "" if args.trueparam is None or args.trueparam == 0 else "_trueparam{}".format(args.trueparam)
        filename = "{}/experiments/data/results/{}_{}{}.npy".format(args.dir, args.modelname, label, trueparam_name)

    elif type_ == "mcmcresults":
        trueparam_name = "" if args.trueparam is None or args.trueparam == 0 else "_trueparam{}".format(args.trueparam)
        chain_name = "_chain{}".format(args.chain) if args.chain > 0 else ""
        filename = "{}/experiments/data/results/{}_{}{}{}.npy".format(args.dir, args.modelname, label, trueparam_name, chain_name)

    elif type_ == "timing":
        filename = "{}/experiments/data/timing/{}_{}_{}_{}_{}_{}{}.npy".format(
            args.dir,
            args.algorithm,
            args.outerlayers,
            args.outertransform,
            "mlp" if args.outercouplingmlp else "resnet",
            args.outercouplinglayers,
            args.outercouplinghidden,
            run_label,
        )
    elif type_ == "paramscan":
        filename = "{}/experiments/data/paramscan/{}.pickle".format(args.dir, args.paramscanstudyname)
    else:
        raise NotImplementedError

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    return filename


def create_modelname(args):
    run_label = "_run{}".format(args.i) if args.i > 0 else ""
    appendix = "" if args.modelname is None else "_" + args.modelname

    try:
        if args.truth:
            if args.dataset in ["spherical_gaussian", "conditional_spherical_gaussian"]:
                args.modelname = "truth_{}_{}_{}_{:.3f}{}{}".format(args.dataset, args.truelatentdim, args.datadim, args.epsilon, appendix, run_label)
            else:
                args.modelname = "truth_{}{}{}".format(args.dataset, appendix, run_label)
            return
    except:
        pass

    if args.dataset in ["spherical_gaussian", "conditional_spherical_gaussian"]:
        args.modelname = "{}{}_{}_{}_{}_{}_{:.3f}{}{}".format(
            args.algorithm,
            "_specified" if args.specified else "",
            args.modellatentdim,
            args.dataset,
            args.truelatentdim,
            args.datadim,
            args.epsilon,
            appendix,
            run_label,
        )
    else:
        args.modelname = "{}{}_{}_{}{}{}".format(args.algorithm, "_specified" if args.specified else "", args.modellatentdim, args.dataset, appendix, run_label)


def nat_to_bit_per_dim(dim):
    if isinstance(dim, (tuple, list, np.ndarray)):
        dim = np.product(dim)
    logger.debug("Nat to bit per dim: factor %s", 1.0 / (np.log(2) * dim))
    return 1.0 / (np.log(2) * dim)


def sum_except_batch(x, num_batch_dims=1):
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)
