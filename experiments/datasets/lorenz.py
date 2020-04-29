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

        self._trajectory = self._lorenz(sigma, beta, rho, np.asarray([x0, y0, z0]), tmax, steps)
        self._x_means=np.array([-0.68170659, -0.55421554, 23.71924285])
        self._x_stds=np.array([7.98203267, 9.0880048 , 8.31979251])

    def is_image(self):
        return False

    def data_dim(self):
        return 3

    def latent_dim(self):
        return 2

    def parameter_dim(self):
        return None

    def trajectory(self):
        x = self._trajectory
        x = self._preprocess(x, inverse=False)
        return x

    def sample(self, n, parameters=None):
        idx = np.random.choice(list(range(len(self._trajectory))), size=n)
        x = self._trajectory[idx, :]
        x = self._preprocess(x, inverse=False)
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

    def _preprocess(self, x, inverse=False):
        x = np.copy(x)
        if self._x_means is not None and self._x_stds is not None:
            if inverse:
                logger.debug("Scaling LHC data back to conventional normalization")
                x *= self._x_stds
                x += self._x_means
            else:
                logger.debug("Scaling LHC data to zero mean and unit variance")
                x = x - self._x_means
                x /= self._x_stds
        else:
            logger.debug("No preprocessing")
        return x
