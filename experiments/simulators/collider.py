import logging

import numpy as np
from scipy.stats import norm, uniform
from manifold_flow.training import NumpyDataset
from experiments.simulators.base import BaseSimulator

logger = logging.getLogger(__name__)


class BaseLHCLoader(BaseSimulator):
    def __init__(self, n_parameters, n_observables, n_final, n_additional_constraints=0, prior_scale=1.0, x_means=None, x_stds=None):
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
            logger.info("Only using %s of %s available samples", limit_samplesize, x.shape[0])
            x = x[:limit_samplesize]
            params = params[:limit_samplesize]

        # Make sure things are sane
        logger.info("ttH features before preprocessing:")
        for i in range(x.shape[1]):
            logger.info("  %s: range %s ... %s, mean %s, std %s", i, np.min(x[:, i]), np.max(x[:, i]), np.mean(x[:, i]), np.std(x[:, i]))

        # Preprocess to zero mean and unit variance
        x = self._preprocess(x)

        # Make sure things are sane
        logger.info("ttH features after preprocessing:")
        for i in range(x.shape[1]):
            logger.info("  %s: range %s ... %s, mean %s, std %s", i, np.min(x[:, i]), np.max(x[:, i]), np.mean(x[:, i]), np.std(x[:, i]))

        # Make sure things are sane
        logger.info("ttH parameters:")
        for i in range(params.shape[1]):
            logger.info("  %s: range %s ... %s, mean %s, std %s", i, np.min(params[:, i]), np.max(params[:, i]), np.mean(params[:, i]), np.std(params[:, i]))

        return NumpyDataset(x, params)

    def default_parameters(self):
        return np.zeros(self._parameter_dim)

    def sample_from_prior(self, n):
        return np.random.normal(
            loc=np.zeros((n, self._parameter_dim)), scale=self._prior_scale * np.ones((n, self._parameter_dim)), size=(n, self._parameter_dim)
        )

    def evaluate_log_prior(self, parameters):
        parameters = parameters.reshape((-1, self.parameter_dim()))
        return np.sum(norm(loc=0.0, scale=self._prior_scale).logpdf(x=parameters), axis=1)

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
    """
    Features:
        0 pt_l1
        1 pt_l2
        2 pt_b1
        3 pt_b2
        4 pt_a1
        5 pt_a2
        6 e_l1
        7 e_l2
        8 e_b1
        9 e_b2
        10 e_a1
        11 e_a2
        12 n_j
        13 n_b
        14 n_l
        15 n_e
        16 n_mu
        17 n_a
        18 eta_l1
        19 eta_l2
        20 eta_b1
        21 eta_b2
        22 eta_a1
        23 eta_a2
        24 phi_l1
        25 phi_l2
        26 phi_b1
        27 phi_b2
        28 phi_a1
        29 phi_a2
        30 met
        31 m_ll
        32 pt_ll
        33 eta_ll
        34 dphi_ll
        35 m_bb
        36 pt_bb
        37 eta_bb
        38 dphi_bb
        39 m_aa
        40 pt_aa
        41 eta_aa
        42 dphi_aa
        43 dphi_l1aa
        44 dphi_l2aa
        45 dphi_b1aa
        46 dphi_b2aa
        47 dphi_METaa
    """
    def __init__(self):
        TTH_X_MEANS = np.asarray(
            [
                8.91563034e01,
                3.80451012e01,
                1.22160637e02,
                6.28824348e01,
                1.48457108e02,
                5.20157127e01,
                1.66163605e02,
                8.18938675e01,
                2.30738586e02,
                1.35192581e02,
                2.74855743e02,
                1.11023781e02,
                3.07111692e00,
                2.06056309e00,
                2.00628996e00,
                8.91026020e-01,
                1.11526406e00,
                2.88845205e00,
                9.74573672e-01,
                1.09504437e00,
                9.96066272e-01,
                1.07848775e00,
                1.01008689e00,
                1.14933527e00,
                7.20288590e-05,
                4.20529163e-03,
                6.18653744e-03,
                1.71405496e-03,
                1.08098506e-03,
                1.53915945e-03,
                1.03206825e02,
                1.23626488e02,
                8.27635422e01,
                3.63268191e-03,
                -4.49752389e-03,
                1.94783340e02,
                1.07005432e02,
                9.47078131e-03,
                -6.46084640e-03,
                1.24853699e02,
                1.70047745e02,
                -2.99490546e-03,
                2.43827514e-03,
                5.80094242e-03,
                -6.84846006e-03,
                3.84141505e-03,
                -6.52486878e-03,
                3.75276082e-03,
            ]
        )
        TTH_X_STDS = np.asarray(
            [
                6.46856537e01,
                3.04017239e01,
                7.57385025e01,
                3.96706085e01,
                1.09678955e02,
                4.37448196e01,
                1.60226028e02,
                9.49661789e01,
                1.99383163e02,
                1.36799286e02,
                2.49974258e02,
                1.15999901e02,
                1.13905418e00,
                2.54389077e-01,
                8.44966769e-02,
                7.00606823e-01,
                7.01320410e-01,
                9.73071277e-01,
                7.12765157e-01,
                8.12319160e-01,
                7.03543305e-01,
                7.67242670e-01,
                7.61939287e-01,
                8.56801689e-01,
                1.81292820e00,
                1.81600308e00,
                1.81281197e00,
                1.81144035e00,
                1.81756973e00,
                1.81763637e00,
                7.61237183e01,
                1.02378113e02,
                6.09270477e01,
                1.59359908e00,
                2.01289272e00,
                1.41503128e02,
                7.44035187e01,
                1.73196638e00,
                2.14188528e00,
                4.28175879e00,
                1.43220505e02,
                1.55024064e00,
                1.78990495e00,
                2.09839201e00,
                1.89159703e00,
                2.19135213e00,
                1.92484295e00,
                2.24098635e00,
            ]
        )
        super().__init__(n_parameters=3, n_observables=48, n_final=8, n_additional_constraints=1, prior_scale=0.5, x_means=TTH_X_MEANS, x_stds=TTH_X_STDS)


class ReducedTopHiggsLoader(BaseLHCLoader):
    def __init__(self):
        TTH_X_MEANS = np.asarray(
            [
                1.24853699e02,  # maa
                1.70047745e02,  # ptaa
            ]
        )
        TTH_X_STDS = np.asarray(
            [
                4.28175879e00,  # maa
                1.43220505e02,  # ptaa
            ]
        )
        super().__init__(n_parameters=3, n_observables=2, n_final=8, n_additional_constraints=1, prior_scale=0.5, x_means=TTH_X_MEANS, x_stds=TTH_X_STDS)
