#! /usr/bin/env python

import numpy as np
from scipy.stats import norm, uniform, special_ortho_group
from sklearn.preprocessing import PolynomialFeatures
import logging
from experiments.simulators.base import BaseSimulator

logger = logging.getLogger(__name__)


class PowerManifoldSimulator(BaseSimulator):
    def __init__(
        self,
        draw_constants=False,
        max_power=5,
        power_decay=3.0,
        base_coeff_std=1.0,
        exp_suppression=0.1,
        weight=0.4,
        const_width=2.0,
        min_width=0.2,
        max_width=1.0,
        filename=None,
    ):
        assert 0.0 < min_width < max_width
        assert power_decay > 0.0
        assert isinstance(max_power, int)
        assert max_power >= 0
        assert 0.0 < weight < 1.0

        self._const_width = const_width
        self._min_width = min_width
        self._max_power = max_power
        self._weight = weight
        self._power_decay = power_decay
        self._max_width = max_width
        self._base_coeff_std = base_coeff_std
        self._exp_suppression = exp_suppression

        if draw_constants:
            self._coeffs, self._rotation = self._draw_constants()
            self._save_constants(filename)
        else:
            self._coeffs, self._rotation = self._load_constants(filename)

    def is_image(self):
        return False

    def data_dim(self):
        return 3

    def latent_dim(self):
        return 2

    def parameter_dim(self):
        return 1

    def sample(self, n, parameters=None):
        assert parameters is not None

        z = self._draw_z(n, parameters=parameters)
        x = self._transform_z_to_x(z)

        return x

    def sample_ood(self, n, parameters=None):
        x = self.sample(n, parameters=np.zeros((n, 1)))
        noise = 0.1 * np.random.normal(size=(n, 3))
        return x + noise

    def log_density(self, x, parameters=None):
        tolerance = 1.0e-3
        z, eps = self._transform_x_to_z(x)
        log_prob_manifold = self._log_density_z(z, parameters=parameters)
        log_prob = np.where(eps.flatten() ** 2 < tolerance ** 2, log_prob_manifold - np.log(2.0 * tolerance), np.zeros_like(log_prob_manifold))
        return log_prob

    def distance_from_manifold(self, x):
        z, eps = self._transform_x_to_z(x)
        return np.sum(eps ** 2, axis=1) ** 0.5

    def sample_from_prior(self, n):
        return uniform.rvs(loc=-1.0, scale=2.0, size=(n, self.parameter_dim()))

    def evaluate_log_prior(self, parameters):
        parameters = parameters.reshape((-1, self.parameter_dim()))
        return np.sum(uniform.logpdf(parameters, loc=-1.0, scale=2.0), axis=1)

    def _draw_constants(self):
        n_terms = (self._max_power + 1) * (self._max_power + 2) // 2
        stddevs = self._base_coeff_std * np.ones(n_terms)
        for i in range(1, self._max_power + 1):
            stddevs[i * (i + 1) // 2 : (i + 1) * (i + 2) // 2] *= float(i) ** (-self._power_decay)
        logger.debug("Stddevs for coefficients: %s", stddevs)

        coeffs = stddevs * np.random.normal(size=n_terms)
        logger.info("Drew new power coefficients: %s", coeffs)

        rot = special_ortho_group.rvs(3)
        rot = np.dot(rot, rot)
        logger.info("Drew new rotation matrix:\n%s", rot)

        return coeffs, rot

    def _load_constants(self, filename):
        container = np.load(filename)
        return container["coeffs"], container["rotation"]

    def _save_constants(self, filename):
        np.savez(filename, coeffs=self._coeffs, rotation=self._rotation)

    def _draw_z(self, n, parameters):
        categories = np.random.choice(2, size=n, replace=True, p=(1.0 - self._weight, self._weight))[:, np.newaxis]

        mean_fix = np.array([[1.0, -1.0]])
        z_fix = mean_fix + self._const_width * np.random.normal(size=(n * 2)).reshape((n, 2))

        mean_var = np.array([[-1.0, 1.0]])
        std_var = self._min_width + (self._max_width - self._min_width) * (0.5 + 0.5 * parameters.reshape((-1, 1)))
        z_var = mean_var + std_var * np.random.normal(size=(n * 2)).reshape((n, 2))

        z = categories * z_var + (1.0 - categories) * z_fix
        logger.info("Latent variables:\n%s", z)
        return z

    def _fz(self, z):
        powers = PolynomialFeatures(self._max_power, include_bias=True, interaction_only=False).fit_transform(z)
        fz = powers.dot(self._coeffs).reshape((-1, 1))
        fz = fz * np.exp(-self._exp_suppression * np.sum(z ** 2, axis=-1))[:, np.newaxis]
        return fz

    def _transform_z_to_x(self, z):
        fz = self._fz(z)
        z_fz = np.concatenate((z, fz), axis=1)
        x = np.einsum("ij,nj->ni", self._rotation, z_fz)
        return x

    def _transform_x_to_z(self, x):
        z_fz = np.einsum("ij,nj->ni", self._rotation.T, x)
        z = z_fz[:, :2]
        offset = z_fz[:, 2, np.newaxis] - self._fz(z)
        return z, offset

    def _log_density_z(self, z, parameters):
        mean_fix = np.array([[1.0, -1.0]])
        z_red = (z - mean_fix) / self._const_width
        logp_fix = np.sum(norm.logpdf(z_red) - np.log(self._const_width, axis=-1))

        mean_var = np.array([[-1.0, 1.0]])
        std_var = self._min_width + (self._max_width - self._min_width) * (0.5 + 0.5 * parameters.reshape((-1, 1)))
        z_red = (z - mean_var) / std_var
        logp_var = np.sum(norm.logpdf(z_red) - np.log(std_var), axis=-1)

        logp = np.log((1.0 - self._weight) * np.exp(logp_fix) + self._weight * np.exp(logp_var))
        return logp
