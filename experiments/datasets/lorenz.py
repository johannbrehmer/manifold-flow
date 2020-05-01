import numpy as np
from scipy.integrate import solve_ivp
import logging

from .base import BaseSimulator, IntractableLikelihoodError

logger = logging.getLogger(__name__)


class LorenzSimulator(BaseSimulator):
    """ Lorenz system, following the conventions on https://en.wikipedia.org/wiki/Lorenz_system """

    def __init__(self, sigma=10.0, beta=8.0 / 3.0, rho=28.0, x0s=None, tmax=1000.0, steps=1000000, warmup=50.0, seed=711, random_trajectories=100):
        assert sigma > 0.0
        assert beta > 0.0
        assert rho > 0.0

        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.warmup = warmup
        self.tmax = tmax
        self.steps = steps

        if x0s is None:
            self.x0s = self._draw_initial_states(random_trajectories, seed)
        else:
            self.x0s = np.asarray(x0s)

        self._trajectories = None
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

    def trajectory(self, i=0):
        self._init_trajectories()
        x = self._trajectories[i]
        x = self._preprocess(x, inverse=False)
        return x

    def sample(self, n, parameters=None):
        self._init_trajectories()

        trajs = np.random.choice(len(self._trajectories), size=n)
        steps = np.random.choice(self.steps, size=n)
        x = self._trajectories[trajs, steps, :]
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
    def _draw_initial_states(n, seed):
        np.random.seed(seed)
        return np.ones((1,3)) + 0.1 * np.random.normal(size=(n, 3))

    def _init_trajectories(self):
        if self._trajectories is None:
            self._trajectories = np.array(
                [
                    self._lorenz(self.sigma, self.beta, self.rho, x0 , self.tmax, self.steps, self.warmup)
                    for x0 in self.x0s
                ]
            )

    @staticmethod
    def _lorenz(sigma, beta, rho, x0, tmax, steps, warmup):
        """ Based on https://en.wikipedia.org/wiki/Lorenz_system#Python_simulation """

        logger.info(f"Solving Lorenz system with sigma = {sigma}, beta = {beta}, rho = {rho}, initial conditions x0 = {x0}, saving {steps} time steps from {warmup} to {tmax}")

        def dxdt(t, x):
            """ Computes x' for a given x """
            return sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]

        ts = np.linspace(warmup, tmax, steps)
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
