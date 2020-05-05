#! /usr/bin/env python

""" Top-level script for hyperparameter tuning (for image data) """

import numpy as np
import logging
import sys
import torch
import configargparse
import copy
import optuna
import pickle

sys.path.append("../")

import train, evaluate
from datasets import load_simulator, load_training_dataset, SIMULATORS, load_test_samples
from training import ForwardTrainer, ConditionalForwardTrainer, losses
from utils import create_filename, create_modelname
from architectures import create_model, ALGORITHMS
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def parse_args():
    parser = configargparse.ArgumentParser()

    # What what what
    parser.add_argument("--algorithm", type=str, default="mf", choices=ALGORITHMS)
    parser.add_argument("--dataset", type=str, default="power", choices=SIMULATORS)
    parser.add_argument("-i", type=int, default=0)

    # Dataset details
    parser.add_argument("--truelatentdim", type=int, default=2)
    parser.add_argument("--datadim", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.01)

    # Fixed model parameters
    parser.add_argument("--modellatentdim", type=int, default=2)
    parser.add_argument("--specified", action="store_true")
    parser.add_argument(
        "--lineartransform",
        type=str,
        default="permutation",
        help="Type of linear transformations inserted between the base transformations: linear, permutation. See Neural Spline Flow paper.",
    )
    parser.add_argument("--levels", type=int, default=3, help="Number of levels in multi-scale architectures for image data (for outer transformation)")
    parser.add_argument("--outertransform", type=str, default="rq-coupling")
    parser.add_argument("--innertransform", type=str, default="rq-coupling")
    parser.add_argument("--outercouplingmlp", action="store_true", help="Use MLP instead of ResNet for coupling layers")
    parser.add_argument("--outercouplinglayers", type=int, default=2, help="Number of layers for coupling layers")
    parser.add_argument("--outercouplinghidden", type=int, default=100)
    parser.add_argument("--conditionalouter", action="store_true")
    parser.add_argument("--pieepsilon", type=float, default=0.01)
    parser.add_argument("--encoderblocks", type=int, default=5)
    parser.add_argument("--encoderhidden", type=int, default=100)
    parser.add_argument("--encodermlp", action="store_true")
    parser.add_argument("--structuredlatents", action="store_true", help="Image data: uses convolutional architecture also for inner transformation h")

    # Fixed training params
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--genbatchsize", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Initial learning rate")
    parser.add_argument("--addnllfactor", type=float, default=0.1)
    parser.add_argument("--nllfactor", type=float, default=1.0)
    parser.add_argument("--sinkhornfactor", type=float, default=10.0)
    parser.add_argument("--samplesize", type=int, default=None)
    parser.add_argument("--doughl1reg", type=float, default=0.0)
    parser.add_argument("--nopretraining", action="store_true")
    parser.add_argument("--noposttraining", action="store_true")
    parser.add_argument("--prepie", action="store_true")
    parser.add_argument("--prepostfraction", type=int, default=3)
    parser.add_argument("--alternate", action="store_true", help="Use alternating training algorithm (e.g. MFMF-MD instead of MFMF-S)")
    parser.add_argument("--sequential", action="store_true", help="Use sequential training algorithm")
    parser.add_argument("--validationsplit", type=float, default=0.25, help="Fraction of train data used for early stopping")
    parser.add_argument("--scandal", type=float, default=None, help="Activates SCANDAL training and sets prefactor of score MSE in loss")

    # Evaluation settings
    parser.add_argument("--generate", type=int, default=1000)
    parser.add_argument("--trueparam", type=int, default=0, help="Index of true parameter point for inference tasks")

    # Hyperparameter optimization
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--metricrecoerrorfactor", type=float, default=100.0)
    parser.add_argument("--metricdistancefactor", type=float, default=1.0)
    parser.add_argument("--paramscanstudyname", type=str, default="paramscan")
    parser.add_argument("--resumestudy", action="store_true")

    # Other settings
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")
    parser.add_argument("--dir", type=str, default="/scratch/jb6504/manifold-flow")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def pick_parameters(args, trial, counter):
    margs = copy.deepcopy(args)

    margs.modelname = "{}_{}".format(args.paramscanstudyname, counter)

    margs.outerlayers = trial.suggest_categorical("outerlayers", [4, 8, 12, 16, 20, 24, 28])
    margs.innerlayers = trial.suggest_int("innerlayers", 3, 20)
    margs.linlayers = trial.suggest_int("linlayers", 1, 3)
    margs.linchannelfactor = trial.suggest_int("linchannelfactor", 1, 2)
    margs.lineartransform = trial.suggest_categorical("lineartransform", ["permutation", "lu", "svd"])
    margs.dropout = trial.suggest_categorical("dropout", [0.0, 0.20])
    margs.splinerange = trial.suggest_categorical("splinerange", [6.0, 8.0, 10.0])
    margs.splinebins = trial.suggest_int("splinebins", 3, 20)
    margs.batchnorm = trial.suggest_categorical("batchnorm", [False, True])
    margs.actnorm = trial.suggest_categorical("actnorm", [False, True])

    margs.batchsize = trial.suggest_categorical("batchsize", [50,])
    margs.msefactor = trial.suggest_loguniform("msefactor", 1.0e-3, 10.)
    margs.uvl2reg = trial.suggest_loguniform("uvl2reg", 1.e-9, 0.1)
    margs.weightdecay = trial.suggest_loguniform("weightdecay", 1.e-9, 0.1)
    margs.clip = trial.suggest_loguniform("clip", 1.0, 100.0)

    create_modelname(margs)

    return margs


if __name__ == "__main__":
    # Logger
    args = parse_args()

    # Output -- silence the normal training output
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO
    )
    if not args.debug:
        for key in logging.Logger.manager.loggerDict:
            if "__main__" not in key and "optuna" not in key:
                logging.getLogger(key).setLevel(logging.info if args.debug else logging.WARNING)

    logger.info("Hi!")
    logger.debug("Starting paramscan.py with arguments %s", args)
    logger.debug("Parameter scan study %s", args.paramscanstudyname)

    counter = -1

    def objective(trial):
        global counter

        counter += 1

        # Hyperparameters
        margs = pick_parameters(args, trial, counter)

        logger.info(f"Starting run {counter} / {args.trials}")
        logger.info(f"Hyperparams:")
        logger.info(f"  outer layers:      {margs.outerlayers}")
        logger.info(f"  linlayers:         {margs.linlayers}")
        logger.info(f"  linchannelfactor:  {margs.linchannelfactor}")
        logger.info(f"  inner layers:      {margs.innerlayers}")
        logger.info(f"  linear transform:  {margs.lineartransform}")
        logger.info(f"  spline range:      {margs.splinerange}")
        logger.info(f"  spline bins:       {margs.splinebins}")
        logger.info(f"  batchnorm:         {margs.batchnorm}")
        logger.info(f"  actnorm:           {margs.actnorm}")
        logger.info(f"  dropout:           {margs.dropout}")
        logger.info(f"  batch size:        {margs.batchsize}")
        logger.info(f"  MSE factor:        {margs.msefactor}")
        logger.info(f"  latent L2 reg:     {margs.uvl2reg}")
        logger.info(f"  weight decay:      {margs.weightdecay}")
        logger.info(f"  gradient clipping: {margs.clip}")

        # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
        torch.multiprocessing.set_start_method("spawn", force=True)

        # Load data
        simulator = load_simulator(margs)
        dataset = load_training_dataset(simulator, margs)

        # Create model
        model = create_model(margs, simulator)

        # Train
        trainer = ForwardTrainer(model) if simulator.parameter_dim() is None else ConditionalForwardTrainer(model)
        common_kwargs, _, _, _ = train.make_training_kwargs(margs, dataset)

        logger.info("Starting training MF: manifold training")
        np.random.seed(123)
        _, val_losses = trainer.train(
            loss_functions=[losses.mse, losses.hiddenl2reg],
            loss_labels=["MSE", "L2_lat"],
            loss_weights=[margs.msefactor, 0.0 if margs.uvl2reg is None else margs.uvl2reg],
            epochs=margs.epochs,
            parameters=(
                list(model.outer_transform.parameters()) + list(model.encoder.parameters()) if args.algorithm == "emf" else model.outer_transform.parameters()
            ),
            forward_kwargs={"mode": "projection", "return_hidden": True},
            **common_kwargs,
        )

        # Save
        torch.save(model.state_dict(), create_filename("model", None, margs))

        # Evaluate reco error
        logger.info("Evaluating reco error")
        model.eval()
        np.random.seed(123)
        x, params = next(iter(trainer.make_dataloader(load_training_dataset(simulator, args), args.validationsplit, 1000, 0)[1]))
        x = x.to(device=trainer.device, dtype=trainer.dtype)
        params = None if simulator.parameter_dim() is None else params.to(device=trainer.device, dtype=trainer.dtype)
        x_reco, _, _ = model(x, context=params, mode="projection")
        reco_error = torch.mean(torch.sum((x - x_reco) ** 2, dim=1) ** 0.5).detach().cpu().numpy()

        # Report results
        logger.info("Results:")
        logger.info("  reco err:     %s", reco_error)

        # Plot reco error
        x = np.clip(np.transpose(x, [0, 2, 3, 1]) / 256.0, 0.0, 1.0)
        x_reco = np.clip(np.transpose(x_reco, [0, 2, 3, 1]) / 256.0, 0.0, 1.0)
        plt.figure(figsize=(6 * 3.0, 5 * 3.0))
        for i in range(15):
            plt.subplot(5, 6, 2 * i + 1)
            plt.imshow(x[i])
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            plt.subplot(5, 6, 2 * i + 2)
            plt.imshow(x_reco[i])
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
        plt.tight_layout()
        filename = create_filename("training_plot", "reco_epoch_A", args)
        plt.savefig(filename.format(i_epoch))

        return reco_error

    # Load saved study object
    if args.resumestudy:
        filename = create_filename("paramscan", None, args)
        logger.info("Loading parameter scan from %s", filename)

        with open(filename, "rb") as file:
            study = pickle.load(file)

    else:
        study = optuna.create_study(study_name=args.paramscanstudyname, direction="minimize")

    # Optimize!
    try:
        study.optimize(objective, n_trials=args.trials)
    except (KeyboardInterrupt, SystemExit):
        logger.warning("Optimization interrupted!")

    # Report best results
    logger.info("Best parameters:")
    for k, v in study.best_params.items():
        logger.info("  %s: %s", k, v)

    # Save result
    filename = create_filename("paramscan", None, args)
    logger.info("Saving parameter scan to %s", filename)

    with open(filename, "wb") as file:
        pickle.dump(study, file)

    logger.info("That's all. Have a nice day!")
