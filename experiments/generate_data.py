#! /usr/bin/env python

import argparse
import os
import sys
import numpy as np
import logging

sys.path.append("../")

from experiments.data_generation import SphericalGaussianSimulator

logger = logging.getLogger(__name__)


def generate(args):
    # Simulator
    if args.dataset == "spherical_gaussian":
        simulator = SphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    else:
        raise NotImplementedError("Unknown dataset {}".format(args.dataset))

    # Sample
    x_train = simulator.sample(args.train)
    x_test = simulator.sample(args.test)

    # Save
    os.mkdirs("{}/data/samples/spherical_gaussian".format(args.dir))
    np.save(
        "{}/data/spherical_gaussian/spherical_gaussian_{}_{}_{}_x_train.npy".format(
            args.dir, args.truelatentdim, args.datadim, args.epsilon
        ),
        x_train,
    )
    np.save(
        "{}/data/samples/spherical_gaussian/spherical_gaussian_{}_{}_{}_x_test.npy".format(
            args.dir, args.truelatentdim, args.datadim, args.epsilon
        ),
        x_test,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="spherical_gaussian", choices=["spherical_gaussian"],)

    parser.add_argument("--truelatentdim", type=int, default=9)
    parser.add_argument("--datadim", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--train", type=int, default=1000000)
    parser.add_argument("--test", type=int, default=10000)

    parser.add_argument("--dir", type=str, default="../")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
        datefmt="%H:%M",
        level=logging.DEBUG if args.debug else logging.INFO,
    )
    logger.info("Hi!")
    generate(args)
    logger.info("All done! Have a nice day!")
