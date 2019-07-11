#! /usr/bin/env python

import numpy as np
import logging
import argparse

logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.INFO,
)


def generate(
        latent_dim=2,
        data_dim=100,
        n_train=1000000,
        n_test=10000,
        base_dir=".",
):
    z_train = np.random.normal(0., 1., size=(latent_dim, n_train))
    z_test = np.random.normal(0., 1., size=(latent_dim, n_test))
    transform = np.random.normal(loc=0., scale=1., size=(data_dim, latent_dim))
    x_train = transform.dot(z_train).T
    x_test = transform.dot(z_test).T

    np.save("{}/data/gaussian/gaussian_{}_{}_x_train.npy".format(base_dir, latent_dim, data_dim), x_train)
    np.save("{}/data/gaussian/gaussian_{}_{}_x_test.npy".format(base_dir, latent_dim, data_dim), x_test)
    np.save("{}/data/gaussian/gaussian_{}_{}_z_train.npy".format(base_dir, latent_dim, data_dim), z_train.T)
    np.save("{}/data/gaussian/gaussian_{}_{}_z_test.npy".format(base_dir, latent_dim, data_dim), z_test.T)
    np.save("{}/data/gaussian/gaussian_{}_{}_transform.npy".format(base_dir, latent_dim, data_dim), transform)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("latent", type=int)
    parser.add_argument("data", type=int)
    parser.add_argument("--dir", type=str, default=".")
    return parser.parse_args()


if __name__ == "__main__":
    logging.info("Hi!")
    args = parse_args()
    generate(args.latent, args.data, base_dir=args.dir)
    logging.info("All done! Have a nice day!")
