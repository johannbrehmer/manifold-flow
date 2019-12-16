import logging

import numpy as np
from scipy.stats import norm, uniform
from manifold_flow.training import NumpyDataset
from experiments.simulators.base import BaseSimulator

logger = logging.getLogger(__name__)


class BaseLHCLoader(BaseSimulator):
    def __init__(self, n_parameters, n_observables, n_final, n_additional_constraints=0, prior_scale=1.):
        super().__init__()

        self._prior_scale = prior_scale
        self._data_dim = n_observables
        self._parameter_dim = n_parameters
        self._latent_dim = self._calculate_collider_latent_dim(n_final, n_additional_constraints)

    def is_image(self):
        return False

    def data_dim(self):
        return self._data_dim

    def latent_dim(self):
        return self._latent_dim

    def parameter_dim(self):
        return self._parameter_dim

    def load_dataset(self, train, dataset_dir):
        data_filename = "{}/x_{}.npy".format(dataset_dir, "train" if train else "test")
        param_filename = "{}/theta_{}.npy".format(dataset_dir, "train" if train else "test")

        return NumpyDataset(data_filename, param_filename)

    def default_parameters(self):
        return np.zeros(self._parameter_dim)

    def sample_from_prior(self, n):
        return np.random.normal(loc=np.zeros((n, self._parameter_dim)), scale=self._prior_scale*np.ones((n, self._parameter_dim)), size=(n, self._parameter_dim))

    def evaluate_log_prior(self, parameters):
        return np.sum(norm(loc=0., scale=self._prior_scale).logpdf(x=parameters.flatten()).reshape(parameters.shape), axis=1)

    @staticmethod
    def _calculate_collider_latent_dim(n_final, n_additional_constraints):
        latent_dim = 3 * n_final  # Four-momenta of final state minus on-shell conditions
        latent_dim -= 3  # Energy-momentum conservation. We now the initial px, py, and have one constraint E_total = pz1 - pz2.
        latent_dim -= n_additional_constraints  # Additional constraints, for instance from intermediate narrow resonances
        return latent_dim


class TopHiggsLoader(BaseLHCLoader):
    def __init__(self):
        super().__init__(n_parameters=3, n_observables=48, n_final=8, n_additional_constraints=1, prior_scale=0.5)
