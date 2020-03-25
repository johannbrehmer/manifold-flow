#! /usr/bin/env python

import numpy as np
import logging
import sys
import torch
import argparse
import copy
import optuna
import pickle

sys.path.append("../")

from experiments import train, evaluate
from experiments.utils.loading import load_training_dataset, load_simulator
from experiments.utils.names import create_filename, create_modelname, ALGORITHMS, SIMULATORS
from experiments.utils.models import create_model

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

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
    parser.add_argument("--outertransform", type=str, default="rq-coupling")
    parser.add_argument("--innertransform", type=str, default="rq-coupling")
    parser.add_argument("--lineartransform", type=str, default="permutation")
    parser.add_argument("--outercouplinghidden", type=int, default=100)
    parser.add_argument("--conditionalouter", action="store_true")
    parser.add_argument("--pieepsilon", type=float, default=0.01)
    parser.add_argument("--encoderblocks", type=int, default=5)
    parser.add_argument("--encoderhidden", type=int, default=100)
    parser.add_argument("--encodermlp", action="store_true")

    # Fixed training params
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--genbatchsize", type=int, default=1000)
    parser.add_argument("--addnllfactor", type=float, default=0.1)
    parser.add_argument("--nllfactor", type=float, default=1.0)
    parser.add_argument("--sinkhornfactor", type=float, default=10.0)
    parser.add_argument("--samplesize", type=int, default=100000)
    parser.add_argument("--doughl1reg", type=float, default=0.0)
    parser.add_argument("--nopretraining", action="store_true")
    parser.add_argument("--noposttraining", action="store_true")
    parser.add_argument("--prepie", action="store_true")
    parser.add_argument("--prepostfraction", type=int, default=3)
    parser.add_argument("--alternate", action="store_false")

    # Evaluation settings
    parser.add_argument("--gridresolution", type=int, default=11)
    parser.add_argument("--generate", type=int, default=1000)

    # Hyperparameter optimization
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--metricnllfactor", type=float, default=1.0)
    parser.add_argument("--metricrecoerrorfactor", type=float, default=100.0)
    parser.add_argument("--metricdistancefactor", type=float, default=10.0)
    parser.add_argument("--paramscanstudyname", type=str, default="paramscan")
    parser.add_argument("--resumestudy", action="store_true")

    # Other settings
    parser.add_argument("--dir", type=str, default="/scratch/jb6504/manifold-flow")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def pick_parameters(args, trial, counter):
    margs = copy.deepcopy(args)

    margs.modelname = "paramscan_{}".format(counter)

    margs.outerlayers = trial.suggest_categorical("outerlayers", [3, 5, 10])
    margs.innerlayers = trial.suggest_categorical("innerlayers", [3, 5, 10])
    margs.outercouplingmlp = trial.suggest_categorical("outercouplingmlp", [False, True])
    margs.outercouplinglayers = trial.suggest_categorical("outercouplinglayers", [1, 2, 3])
    margs.dropout = trial.suggest_categorical("dropout", [0.0, 0.20])
    margs.splinerange = trial.suggest_categorical("splinerange", [5.0, 6.0, 8.0])
    margs.splinebins = trial.suggest_categorical("splinebins", [5, 10, 20])

    margs.batchsize = trial.suggest_categorical("batchsize", [50, 100, 200, 500])
    margs.lr = trial.suggest_loguniform("lr", 1.0e-5, 1.0e-2)
    margs.msefactor = trial.suggest_loguniform("msefactor", 1.0e2, 1.0e4)
    margs.weightdecay = trial.suggest_loguniform("weightdecay", 1.0e-8, 1.0e-4)
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
    logger.info("Starting paramscan.py with arguments %s", args)
    logger.debug("Parameter scan study %s", args.paramscanstudyname)

    counter = -1

    def objective(trial):
        global counter

        counter += 1

        # Hyperparameters
        margs = pick_parameters(args, trial, counter)

        logger.info("Starting training for the following hyperparameters:")
        for k, v in margs.__dict__.items():
            logger.info("  %s: %s", k, v)

        # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
        torch.multiprocessing.set_start_method("spawn", force=True)

        # Load data
        simulator = load_simulator(margs)
        dataset = load_training_dataset(simulator, margs)

        # Create model
        model = create_model(margs, simulator)

        # Train
        _ = train.train_model(margs, dataset, model, simulator)

        # Save
        torch.save(model.state_dict(), create_filename("model", None, margs))

        # Evaluate
        model.eval()

        # Evaluate test samples
        log_likelihood_test, reconstruction_error_test, _ = evaluate.evaluate_test_samples(margs, simulator, model, paramscan=True)
        mean_log_likelihood_test = np.mean(log_likelihood_test)
        mean_reco_error_test = np.mean(reconstruction_error_test)

        # Generate samples
        x_gen = evaluate.sample_from_model(margs, model, simulator)
        distances_gen = simulator.distance_from_manifold(x_gen)
        mean_gen_distance = np.mean(distances_gen)

        # Report results
        logger.info("Results:")
        logger.info("  test log p:    %s", mean_log_likelihood_test)
        logger.info("  test reco err: %s", mean_reco_error_test)
        logger.info("  gen distance:  %s", mean_gen_distance)

        return (
            -1.0 * margs.metricnllfactor * mean_log_likelihood_test
            + margs.metricrecoerrorfactor * mean_reco_error_test
            + margs.metricdistancefactor * mean_gen_distance
        )

    # Load saved study object
    if args.resumestudy:
        filename = create_filename("paramscan", None, args)
        logger.info("Loading parameter scan from %s", filename)

        with open(filename, "rb") as file:
            study = pickle.load(file)

    # Optimize!
    study = optuna.create_study(study_name=args.paramscanstudyname, direction="minimize")
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
