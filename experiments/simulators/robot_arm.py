#! /usr/bin/env python

import numpy as np
from scipy.stats import norm, uniform
import logging
from experiments.simulators.base import BaseSimulator, IntractableLikelihoodError

logger = logging.getLogger(__name__)


class RobotArmSimulator(BaseSimulator):
    def __init__(self, l1=1.0, l2=0.5, base_mean=0.0, base_std=1.0, phi1_mean=0.25 * np.pi, phi1_std=0.125 * np.pi, phi2_std=0.125 * np.pi, noise=0.0):
        self.l1 = l1
        self.l2 = l2
        self.d_mean = base_mean
        self.d_std = base_std
        self.phi1_mean = phi1_mean
        self.phi1_std = phi1_std
        self.phi2_std = phi2_std
        self.noise = noise

    def is_image(self):
        return False

    def data_dim(self):
        return 6

    def latent_dim(self):
        return 3

    def parameter_dim(self):
        return 1

    def log_density(self, x, parameters=None):
        raise IntractableLikelihoodError

    def sample(self, n, parameters=None):
        assert parameters is not None
        z = self._draw_z(n, parameters=parameters)
        x = self._observation(z)
        return x

    def distance_from_manifold(self, x):
        raise IntractableLikelihoodError

    def sample_from_prior(self, n):
        return norm.rvs(size=(n, self.parameter_dim()))

    def evaluate_log_prior(self, parameters):
        parameters = parameters.reshape((-1, self.parameter_dim()))
        return np.sum(norm.logpdf(parameters), axis=1)

    def _draw_z(self, n, parameters):
        d = np.random.normal(self.d_mean, self.d_std, size=(n, 1))
        phi1 = np.random.normal(self.phi1_mean, self.phi1_std, size=(n, 1))
        phi2 = np.random.normal(parameters, self.phi2_std, size=(n, 1))
        z = np.concatenate((d, phi1, phi2), axis=1)
        return z

    def _observation(self, z):
        n = z.shape[0]
        d = z[:, 0]
        phi1 = z[:, 0]
        phi2 = z[:, 0]

        x0 = np.concatenate((d.reshape(-1, 1), np.zeros((n, 1))), axis=1)
        x1 = x0 + self.l1 * np.concatenate((np.cos(phi1).reshape((-1, 1)), np.sin(phi1).reshape((-1, 1))), axis=1)
        x2 = x1 + self.l2 * np.concatenate((np.cos(phi1 + phi2).reshape((-1, 1)), np.sin(phi1 + phi2).reshape((-1, 1))), axis=1)
        x = np.concatenate((x0, x1, x2), axis=1)

        if self.noise > 0.0:
            x += np.random.normal(0.0, self.noise, size=(n, 6))

        return x
