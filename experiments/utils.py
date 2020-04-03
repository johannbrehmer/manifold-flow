import os


def create_filename(type, label, args):
    run_label = "_run{}".format(args.i) if args.i > 0 else ""

    if type == "dataset":  # Fixed datasets
        filename = "{}/experiments/data/samples/{}".format(args.dir, args.dataset)

    elif type == "sample":  # Dynamically sampled from simulator
        if args.dataset in ["spherical_gaussian", "conditional_spherical_gaussian"]:
            filename = "{}/experiments/data/samples/{}/{}_{}_{}_{:.3f}_{}{}.npy".format(
                args.dir, args.dataset, args.dataset, args.truelatentdim, args.datadim, args.epsilon, label, run_label
            )
        else:
            filename = "{}/experiments/data/samples/{}/{}{}.npy".format(args.dir, args.dataset, label, run_label)

    elif type == "model":
        filename = "{}/experiments/data/models/{}.pt".format(args.dir, args.modelname)

    elif type == "checkpoint":
        filename = "{}/experiments/data/models/checkpoints/{}.pt".format(args.dir, args.modelname)

    elif type == "training_plot":
        filename = "{}/experiments/data/figures/training/{}_epoch{}.pdf".format(args.dir, args.modelname, "{}")

    elif type == "learning_curve":
        filename = "{}/experiments/data/learning_curves/{}.npy".format(args.dir, args.modelname)

    elif type == "results":
        trueparam_name = "_trueparam{}".format(args.trueparam) if args.trueparam > 0 else ""
        filename = "{}/experiments/data/results/{}_{}{}.npy".format(args.dir, args.modelname, label, trueparam_name)

    elif type == "mcmcresults":
        trueparam_name = "_trueparam{}".format(args.trueparam) if args.trueparam > 0 else ""
        chain_name = "_chain{}".format(args.chain) if args.chain > 0 else ""
        filename = "{}/experiments/data/results/{}_{}{}{}.npy".format(args.dir, args.modelname, label, trueparam_name, chain_name)

    elif type == "timing":
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
    elif type == "paramscan":
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
