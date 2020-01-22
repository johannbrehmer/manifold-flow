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
                94.40898, 38.293396, 129.17973, 63.718967, 181.19, 66.270294, 162.78603, 76.85954, 227.52643, 129.116, 313.12534, 129.18875, 0.9078872, 1.0456945, 0.93022776,
                1.0281563, 0.9229034, 1.065798, -1.1027638e-05, 0.00403136, 0.005703883, 0.006515818, 0.0029550078, 2.660957e-05, 115.53658, 120.22046, 90.383385, 0.010395338,
                -0.012373894, 191.4471, 119.31814, 0.017556058, -0.008351761, 124.96437, 219.41402, -0.0082995165, -0.004001956, 0.0052848016, -0.013329979, 0.0075188284,
                -0.0065666684, 0.0036475079
            ]
        )
        TTH_X_STDS = np.asarray(
            [
                56.933517, 29.633202, 66.08287, 37.361572, 73.54688, 26.476595, 168.53899, 100.35491, 193.60933, 143.58432, 220.26591, 102.159164, 0.7301178, 0.8286062, 0.7238323,
                0.798472, 0.81572974, 0.9066472, 1.8007879, 1.8169788, 1.8021392, 1.8071805, 1.8182082, 1.8168657, 59.218544, 104.02254, 49.477535, 1.694322, 2.0485785, 141.08936,
                59.68141, 1.8453279, 2.202337, 4.7465997, 89.33402, 1.6982772, 1.9781849, 2.0325553, 1.8577149, 2.12811, 1.8756281, 2.136366,
            ]
        )
        super().__init__(n_parameters=3, n_observables=42, n_final=8, n_additional_constraints=1, prior_scale=10., x_means=TTH_X_MEANS, x_stds=TTH_X_STDS)


class ReducedTopHiggsLoader(BaseLHCLoader):
    def __init__(self):
        TTH_X_MEANS = np.asarray([124.96437, 219.41402])  # maa  # ptaa
        TTH_X_STDS = np.asarray([4.7465997, 89.33402])  # maa  # ptaa
        super().__init__(n_parameters=3, n_observables=2, n_final=8, n_additional_constraints=1, prior_scale=10., x_means=TTH_X_MEANS, x_stds=TTH_X_STDS)
