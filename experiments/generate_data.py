#! /usr/bin/env python

import argparse
import os
import sys
import numpy as np
import logging

sys.path.append("../")

from experiments.utils import _load_simulator, _filename

logger = logging.getLogger(__name__)


def generate(args):
    # Simulator
    simulator = _load_simulator(args)

    # Parameters?
    conditional = simulator.parameter_dim() is not None

    parameters_train = simulator.sample_from_prior(args.train) if conditional else None
    parameters_test = np.vstack([simulator.default_parameters() for _ in range(args.test)])


    # Sample
    logger.info("Generating %s training samples at parameters %s", args.train, parameters_train)
    x_train = simulator.sample(args.train, parameters=parameters_train)

    logger.info("Generating %s test samples at parameters %s", args.test, parameters_test)
    x_test = simulator.sample(args.test, parameters=parameters_test)

    # Save
    np.save(_filename("sample", "x_train", args), x_train)
    np.save(_filename("sample", "x_test", args), x_test)
    if conditional:
        np.save(_filename("sample", "parameters_train", args), parameters_train)
        np.save(_filename("sample", "parameters_test", args), parameters_test)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="spherical_gaussian", choices=["spherical_gaussian", "conditional_spherical_gaussian"])

    parser.add_argument("--truelatentdim", type=int, default=10)
    parser.add_argument("--datadim", type=int, default=15)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--train", type=int, default=1000000)
    parser.add_argument("--test", type=int, default=10000)

    parser.add_argument("--dir", type=str, default="../")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO
    )
    logger.info("Hi!")
    generate(args)
    logger.info("All done! Have a nice day!")
