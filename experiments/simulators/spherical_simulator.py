#! /usr/bin/env python

import numpy as np
from scipy.stats import norm
import itertools
import logging
from experiments.simulators.base import BaseSimulator

logger = logging.getLogger(__name__)


class SphericalGaussianSimulator(BaseSimulator):
    def __init__(self, latent_dim=8, data_dim=9, phases=0.5 * np.pi, widths=0.25 * np.pi, epsilon=0.01):
        self._latent_dim = latent_dim
        self._data_dim = data_dim
        self._phases = phases * np.ones(latent_dim) if isinstance(phases, float) else phases
        self._widths = widths * np.ones(latent_dim) if isinstance(widths, float) else widths
        self._epsilon = epsilon

        assert data_dim > latent_dim
        assert epsilon > 0.0

    def is_image(self):
        return False

    def data_dim(self):
        return self._data_dim

    def latent_dim(self):
        return self._latent_dim

    def parameter_dim(self):
        return None

    def log_density(self, x, parameters=None, precise=False):
        z_phi, z_eps = self._transform_x_to_z(x)
        logp = self._log_density(z_phi, z_eps, precise=precise)
        return logp

    def sample(self, n, parameters=None):
        z_phi, z_eps = self._draw_z(n)
        x = self._transform_z_to_x(z_phi, z_eps)
        return x

    def sample_ood(self, n, parameters=None):
        z_phi, _ = self._draw_z(n)
        z_eps = np.random.uniform(-0.1, 0.1, size=(n, self._data_dim - self._latent_dim))
        x = self._transform_z_to_x(z_phi, z_eps)
        return x

    def distance_from_manifold(self, x):
        z_phi, z_eps = self._transform_x_to_z(x)
        return np.sum(z_eps ** 2, axis=1) ** 0.5

    def _draw_z(self, n):
        # Spherical coordinates
        phases_ = np.empty((n, self._latent_dim))
        phases_[:] = self._phases
        widths_ = np.empty((n, self._latent_dim))
        widths_[:] = self._widths
        z_phi = np.random.normal(phases_, widths_, size=(n, self._latent_dim))
        z_phi = np.mod(z_phi, 2.0 * np.pi)

        # Fuzzy coordinates
        z_eps = np.random.normal(0.0, self._epsilon, size=(n, self._data_dim - self._latent_dim))
        return z_phi, z_eps

    def _transform_z_to_x(self, z_phi, z_eps):
        r = 1.0 + z_eps[:, 0]
        a = np.concatenate((2 * np.pi * np.ones((z_phi.shape[0], 1)), z_phi), axis=1)  # n entries, each (2 pi, z_sub)
        sins = np.sin(a)
        sins[:, 0] = 1
        sins = np.cumprod(sins, axis=1)  # n entries, each (1, sin(z0), sin(z1), ..., sin(zk))
        coss = np.cos(a)
        coss = np.roll(coss, -1, axis=1)  # n entries, each (cos(z0), cos(z1), ..., cos(zk), 1)
        exact_sphere = sins * coss  # (n, k+1)
        fuzzy_sphere = exact_sphere * r[:, np.newaxis]
        x = np.concatenate((fuzzy_sphere, z_eps[:, 1:]), axis=1)
        return x

    def _transform_x_to_z(self, x):
        z_phi = np.zeros((x.shape[0], self._latent_dim))
        for i in range(self._latent_dim):
            z_phi[:, i] = np.arccos(x[:, i] / np.sum(x[:, i : self._latent_dim + 1] ** 2, axis=1) ** 0.5)
        # Special case for last component, see https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
        z_phi[:, self._latent_dim - 1] = np.where(x[:, self._latent_dim] < 0.0, 2.0 * np.pi - z_phi[:, self._latent_dim - 1], z_phi[:, self._latent_dim - 1])

        r = np.sum(x[:, : self._latent_dim + 1] ** 2, axis=1) ** 0.5
        z_eps = np.copy(x[:, self._latent_dim :])
        z_eps[:, 0] = r - 1
        return z_phi, z_eps

    def _log_density(self, z_phi, z_eps, precise=False):
        r = 1.0 + z_eps[:, 0]
        phases_ = np.empty((z_phi.shape[0], self._latent_dim))
        phases_[:] = self._phases
        widths_ = np.empty((z_phi.shape[0], self._latent_dim))
        widths_[:] = self._widths

        p_sub = 0.0
        for shifted_z_phi in self._generate_equivalent_coordinates(z_phi, precise):
            p_sub += norm(loc=phases_, scale=widths_).pdf(shifted_z_phi)

        logp_sub = np.log(p_sub)
        logp_eps = np.log(norm(loc=0.0, scale=self._epsilon).pdf(z_eps))

        log_det = self._latent_dim * np.log(np.abs(r))
        log_det += np.sum(np.arange(self._latent_dim - 1, -1, -1)[np.newaxis, :] * np.log(np.abs(np.sin(z_phi))), axis=1)

        logp = np.sum(logp_sub, axis=1) + np.sum(logp_eps, axis=1) + log_det
        return logp

    def _generate_equivalent_coordinates(self, z_phi, precise):
        # Restrict z to canonical range: [0, pi) for polar angles, and [0., 2pi) for azimuthal angle
        z_phi = z_phi % (2.0 * np.pi)
        z_phi[:, :-1] = np.where(z_phi[:, :-1] > np.pi, 2.0 * np.pi - z_phi[:, :-1], z_phi[:, :-1])

        # Yield this one
        yield z_phi

        # Variations of polar angles
        for dim in range(self._latent_dim - 1):
            z = np.copy(z_phi)
            z[:, dim] = -z_phi[:, dim]
            yield z

            z = np.copy(z_phi)
            z[:, dim] = 2.0 * np.pi - z_phi[:, dim]
            yield z

        # Variations of aximuthal angle
        z = np.copy(z_phi)
        z[:, -1] = -2.0 * np.pi + z_phi[:, -1]
        yield z

        z = np.copy(z_phi)
        z[:, -1] = 2.0 * np.pi + z_phi[:, -1]
        yield z
