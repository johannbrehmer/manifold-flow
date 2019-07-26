#! /usr/bin/env python

import numpy as np
import logging
import argparse
import os
from scipy.stats import norm


def simulator(epsilon, latent_dim, data_dim, n, phases, widths):
    z_phi, z_eps = _draw_z(epsilon, latent_dim, data_dim, n, phases, widths)
    x = _transform_z_to_x(z_phi, z_eps)
    return x


def true_logp(x, epsilon, latent_dim, phases, widths):
    z_phi, z_eps = _transform_x_to_z(x, latent_dim)
    logp = _log_likelihood(z_phi, z_eps, latent_dim, phases, widths, epsilon)
    return logp


def _draw_z(epsilon, latent_dim, data_dim, n, phases, widths):
    # Spherical coordinates
    phases_ = np.empty((n, latent_dim))
    phases_[:] = phases
    widths_ = np.empty((n, latent_dim))
    widths_[:] = widths
    z_phi = np.random.normal(phases_, widths_, size=(n, latent_dim))
    z_phi = np.mod(z_phi, 2.0 * np.pi)

    # Fuzzy coordinates
    z_eps = np.random.normal(0.0, epsilon, size=(n, data_dim - latent_dim))
    return z_phi, z_eps


def _transform_z_to_x(z_phi, z_eps):
    r = 1.0 + z_eps[:, 0]
    a = np.concatenate(
        (2 * np.pi * np.ones((z_phi.shape[0], 1)), z_phi), axis=1
    )  # n entries, each (2 pi, z_sub)
    sins = np.sin(a)
    sins[:, 0] = 1
    sins = np.cumprod(
        sins, axis=1
    )  # n entries, each (1, sin(z0), sin(z1), ..., sin(zk))
    coss = np.cos(a)
    coss = np.roll(coss, -1)  # n entries, each (cos(z0), cos(z1), ..., cos(zk), 1)
    exact_sphere = sins * coss  # (n, k+1)
    fuzzy_sphere = exact_sphere * r[:, np.newaxis]
    x = np.concatenate((fuzzy_sphere, z_eps[:, 1:]), axis=1)
    return x


def _transform_x_to_z(x, latent_dim):
    z_phi = np.zeros((x.shape[0], latent_dim))
    for i in range(latent_dim):
        z_phi[:, i] = np.arccos(
            x[:, i] / np.sum(x[:, i : latent_dim + 1] ** 2, axis=1) ** 0.5
        )
    r = np.sum(x[:, : latent_dim + 1] ** 2, axis=1) ** 0.5
    z_eps = x[:, latent_dim:]
    z_eps[:, 0] = r - 1
    return z_phi, z_eps


def _log_likelihood(z_phi, z_eps, latent_dim, phases, widths, epsilon):
    r = z_eps[:, 0]
    phases_ = np.empty((z_phi.shape[0], latent_dim))
    phases_[:] = phases
    widths_ = np.empty((z_phi.shape[0], latent_dim))
    widths_[:] = widths

    logp_sub = np.log(norm(loc=phases_, scale=widths_).pdf(z_phi))
    logp_eps = np.log(norm(loc=0.0, scale=epsilon).pdf(z_eps))

    log_det = latent_dim * np.abs(r)
    log_det += np.sum(
        np.arange(latent_dim - 1, 0, -1)[np.newaxis, :] * np.log(np.abs(np.sin(z_phi))),
        axis=1,
    )

    logp = np.concatenate((logp_sub, logp_eps), axis=1)
    logp = np.sum(logp, axis=1) + log_det
    return logp


def generate(
    epsilon,
    latent_dim=15,
    data_dims=[16, 32, 64, 128],
    n_train=1000000,
    n_test=10000,
    base_dir=".",
):
    if not os.path.exists("{}/data/spherical_gaussian".format(base_dir)):
        os.mkdir("{}/data/spherical_gaussian".format(base_dir))
    phases = np.random.uniform(low=0.0, high=2.0 * np.pi, size=latent_dim)
    widths = np.random.uniform(low=0.5, high=2.0, size=latent_dim)
    np.save(
        "{}/data/spherical_gaussian/spherical_gaussian_phases.npy".format(base_dir),
        phases,
    )
    np.save(
        "{}/data/spherical_gaussian/spherical_gaussian_widths.npy".format(base_dir),
        widths,
    )

    for data_dim in data_dims:
        x_train = simulator(epsilon, latent_dim, data_dim, n_train, phases, widths)
        x_test = simulator(epsilon, latent_dim, data_dim, n_test, phases, widths)
        np.save(
            "{}/data/spherical_gaussian/spherical_gaussian_{}_{}_x_train.npy".format(
                base_dir, latent_dim, data_dim
            ),
            x_train,
        )
        np.save(
            "{}/data/spherical_gaussian/spherical_gaussian_{}_{}_x_test.npy".format(
                base_dir, latent_dim, data_dim
            ),
            x_test,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument(
        "--dir",
        type=str,
        default="/Users/johannbrehmer/work/projects/ae_flow/autoencoded-flow",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
        datefmt="%H:%M",
        level=logging.INFO,
    )
    logging.info("Hi!")
    args = parse_args()
    generate(args.epsilon, base_dir=args.dir)
    logging.info("All done! Have a nice day!")
