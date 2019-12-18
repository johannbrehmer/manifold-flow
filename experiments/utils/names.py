import os

SIMULATORS = ["spherical_gaussian", "conditional_spherical_gaussian", "tth", "cifar10", "imagenet"]
ALGORITHMS = ["flow", "pie", "mf", "slice", "gamf", "hybrid"]


def create_filename(type, label, args):
    if type == "dataset":  # Fixed datasets
        filename = "{}/experiments/data/samples/{}".format(args.dir, args.dataset)
    elif type == "sample":  # Dynamically sampled from simulator
        if args.dataset in ["spherical_gaussian", "conditional_spherical_gaussian"]:
            filename = "{}/experiments/data/samples/{}/{}_{}_{}_{:.3f}_{}.npy".format(
                args.dir, args.dataset, args.dataset, args.truelatentdim, args.datadim, args.epsilon, label
            )
        else:
            filename = "{}/experiments/data/samples/{}/{}.npy".format(
                args.dir, args.dataset, label
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
    appendix = "" if args.modelname is None else "_" + args.modelname
    if args.dataset in ["spherical_gaussian", "conditional_spherical_gaussian"]:
        args.modelname = "{}_{}_{}_{}_{}_{:.3f}{}".format(
            args.algorithm, args.modellatentdim, args.dataset, args.truelatentdim, args.datadim, args.epsilon, appendix
        )
    else:
        args.modelname = "{}_{}_{}{}".format(args.algorithm, args.modellatentdim, args.dataset, appendix)
