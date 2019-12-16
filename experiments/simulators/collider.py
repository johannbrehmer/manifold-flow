import logging

import numpy as np
from scipy.stats import norm, uniform
from manifold_flow.training import NumpyDataset
from experiments.simulators.base import BaseSimulator

logger = logging.getLogger(__name__)


class BaseLHCLoader(BaseSimulator):
    def __init__(self, n_parameters, n_observables, n_final, n_additional_constraints=0, prior_scale=1., x_means=None, x_stds=None):
        super().__init__()

        self._prior_scale = prior_scale
        self._data_dim = n_observables
        self._parameter_dim = n_parameters
        self._latent_dim = self._calculate_collider_latent_dim(n_final, n_additional_constraints)
        self._x_means = x_means
        self._x_stds = x_stds

    def is_image(self):
        return False

    def data_dim(self):
        return self._data_dim

    def latent_dim(self):
        return self._latent_dim

    def parameter_dim(self):
        return self._parameter_dim

    def load_dataset(self, train, dataset_dir, limit_samplesize=None):
        # Load numpy arrays
        x = np.load("{}/x_{}.npy".format(dataset_dir, "train" if train else "test"))
        params = np.load("{}/theta_{}.npy".format(dataset_dir, "train" if train else "test"))

        # OPtionally limit sample size
        if limit_samplesize is not None:
            x = x[:limit_samplesize]
            params = params[:limit_samplesize]

        # Preprocess to zero mean and unit variance
        x = self._preprocess(x)

        return NumpyDataset(x, params)

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

    def _preprocess(self, x):
        if self._x_means is not None and self._x_stds is not None:
            x = x - self._x_means
            x /= self._x_stds
        return x


class TopHiggsLoader(BaseLHCLoader):
    def __init__(self):
        TTH_X_MEANS = np.asarray([
            8.91563034e+01,  3.80451012e+01,  1.22160637e+02,  6.28824348e+01,
            1.48457108e+02,  5.20157127e+01,  1.66163605e+02,  8.18938675e+01,
            2.30738586e+02,  1.35192581e+02,  2.74855743e+02,  1.11023781e+02,
            3.07111692e+00,  2.06056309e+00,  2.00628996e+00,  8.91026020e-01,
            1.11526406e+00,  2.88845205e+00,  9.74573672e-01,  1.09504437e+00,
            9.96066272e-01,  1.07848775e+00,  1.01008689e+00,  1.14933527e+00,
            7.20288590e-05,  4.20529163e-03,  6.18653744e-03,  1.71405496e-03,
            1.08098506e-03,  1.53915945e-03,  1.03206825e+02,  1.23626488e+02,
            8.27635422e+01,  3.63268191e-03, -4.49752389e-03,  1.94783340e+02,
            1.07005432e+02,  9.47078131e-03, -6.46084640e-03,  1.24853699e+02,
            1.70047745e+02, -2.99490546e-03,  2.43827514e-03,  5.80094242e-03,
           -6.84846006e-03,  3.84141505e-03, -6.52486878e-03,  3.75276082e-03
        ])
        TTH_X_STDS = np.asarray([
            6.46856537e+01, 3.04017239e+01, 7.57385025e+01, 3.96706085e+01,
            1.09678955e+02, 4.37448196e+01, 1.60226028e+02, 9.49661789e+01,
            1.99383163e+02, 1.36799286e+02, 2.49974258e+02, 1.15999901e+02,
            1.13905418e+00, 2.54389077e-01, 8.44966769e-02, 7.00606823e-01,
            7.01320410e-01, 9.73071277e-01, 7.12765157e-01, 8.12319160e-01,
            7.03543305e-01, 7.67242670e-01, 7.61939287e-01, 8.56801689e-01,
            1.81292820e+00, 1.81600308e+00, 1.81281197e+00, 1.81144035e+00,
            1.81756973e+00, 1.81763637e+00, 7.61237183e+01, 1.02378113e+02,
            6.09270477e+01, 1.59359908e+00, 2.01289272e+00, 1.41503128e+02,
            7.44035187e+01, 1.73196638e+00, 2.14188528e+00, 4.28175879e+00,
            1.43220505e+02, 1.55024064e+00, 1.78990495e+00, 2.09839201e+00,
            1.89159703e+00, 2.19135213e+00, 1.92484295e+00, 2.24098635e+00
        ])
        super().__init__(n_parameters=3, n_observables=48, n_final=8, n_additional_constraints=1, prior_scale=0.5, x_means=TTH_X_MEANS, x_stds=TTH_X_STDS)
