import numpy as np
from scipy.integrate import solve_ivp
import logging

from .base import BaseSimulator, IntractableLikelihoodError

logger = logging.getLogger(__name__)


class LorenzSimulator(BaseSimulator):
    """ Lorenz system, following the conventions on https://en.wikipedia.org/wiki/Lorenz_system """

    def __init__(self, sigma=10.0, beta=8.0 / 3.0, rho=28.0, x0=1.0, y0=1.0, z0=1.0, tmax=100.0, steps=10000000):
        assert sigma > 0.0
        assert beta > 0.0
        assert rho > 0.0

        self.trajectory = self._lorenz(sigma, beta, rho, np.asarray([x0, y0, z0]), tmax, steps)

    def is_image(self):
        return False

    def data_dim(self):
        return 3

    def latent_dim(self):
        return 2

    def parameter_dim(self):
        return None

    def sample(self, n, parameters=None):
        idx = np.random.choice(list(range(len(self.trajectory))), size=n)
        x = self.trajectory[idx, :]
        return x

    def sample_ood(self, n, parameters=None):
        x = self.sample(n, parameters=np.zeros((n, 1)))
        noise = 0.1 * np.random.normal(size=(n, 3))
        return x + noise

    def log_density(self, x, parameters=None):
        raise IntractableLikelihoodError

    def distance_from_manifold(self, x):
        raise NotImplementedError

    @staticmethod
    def _lorenz(sigma, beta, rho, x0, tmax, steps):
        """ Based on https://en.wikipedia.org/wiki/Lorenz_system#Python_simulation """

        logger.info(f"Solving Lorenz system with parameters sigma = {sigma}, beta = {beta}, rho = {rho}, initial conditions x0 = {x0}, for {steps} time steps from 0 to {tmax}")

        def dxdt(t, x):
            """ Computes x' for a given x """
            return sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]

        ts = np.linspace(0.0, tmax, steps)
        results = solve_ivp(fun=dxdt, y0=x0, t_span=[0., tmax], t_eval=ts)
        xs = results.y.T
        logger.debug("Done")
        return xs

    def sample_from_prior(self, n):
        raise NotImplementedError

    def evaluate_log_prior(self, parameters):
        raise NotImplementedError
