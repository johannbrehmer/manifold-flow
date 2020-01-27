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
        logger.info("lhc features before preprocessing:")
        for i in range(x.shape[1]):
            logger.info("  %s: range %s ... %s, mean %s, std %s", i, np.min(x[:, i]), np.max(x[:, i]), np.mean(x[:, i]), np.std(x[:, i]))

        # Preprocess to zero mean and unit variance
        x = self._preprocess(x)

        # Make sure things are sane
        logger.info("lhc features after preprocessing:")
        for i in range(x.shape[1]):
            logger.info("  %s: range %s ... %s, mean %s, std %s", i, np.min(x[:, i]), np.max(x[:, i]), np.mean(x[:, i]), np.std(x[:, i]))

        # Make sure things are sane
        logger.info("lhc parameters:")
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


class WBFLoader(BaseLHCLoader):
    """
    Features:
    0 e_a1
    1 px_a1
    2 py_a1
    3 pz_a1
    4 pt_a1
    5 eta_a1
    6 phi_a1
    7 e_a2
    8 px_a2
    9 py_a2
    10 pz_a2
    11 pt_a2
    12 eta_a2
    13 phi_a2
    14 e_j1
    15 px_j1
    16 py_j1
    17 pz_j1
    18 pt_j1
    19 eta_j1
    20 phi_j1
    21 e_j2
    22 px_j2
    23 py_j2
    24 pz_j2
    25 pt_j2
    26 eta_j2
    27 phi_j2
    28 e_aa
    29 px_aa
    30 py_aa
    31 pz_aa
    32 pt_aa
    33 m_aa
    34 eta_aa
    35 phi_aa
    36 deltaeta_aa
    37 deltaphi_aa
    38 e_jj
    39 px_jj
    40 py_jj
    41 pz_jj
    42 pt_jj
    43 m_jj
    44 eta_jj
    45 phi_jj
    46 deltaeta_jj
    47 deltaphi_jj
    """

    def __init__(self):
        X_MEANS = np.array(
            [
                399.36316,
                -0.4373968,
                0.63366896,
                -0.63484854,
                249.49168,
                -0.003057833,
                0.0021281342,
                131.79462,
                -0.058283027,
                0.24817096,
                -0.76308465,
                76.3759,
                -0.0058261123,
                0.014732182,
                757.3401,
                -0.10769362,
                -0.60862094,
                -6.1785574,
                264.5138,
                -0.008442465,
                -0.0077702953,
                574.56604,
                0.077811696,
                -0.0940453,
                2.8334484,
                104.09844,
                0.0012077185,
                -0.002446026,
                531.15753,
                -0.49569345,
                0.8818537,
                -1.3979495,
                304.61646,
                128.13852,
                -0.0041540307,
                0.0058416715,
                0.002768248,
                -0.012604275,
                1331.9509,
                -0.029888684,
                -0.70272374,
                -3.345049,
                285.39468,
                884.0687,
                -0.008315945,
                -0.0039043655,
                -0.00965049,
                -0.0053242273,
            ]
        )
        X_STDS = np.array(
            [
                299.27292,
                211.80585,
                211.97644,
                398.90625,
                166.07448,
                1.0559034,
                1.815807,
                111.966324,
                64.14647,
                64.06791,
                147.19014,
                48.890263,
                1.1151956,
                1.8119942,
                634.10254,
                222.78296,
                223.05367,
                935.3005,
                171.6325,
                1.7551638,
                1.8129121,
                597.7141,
                90.115036,
                90.215744,
                818.73914,
                73.66688,
                2.31682,
                1.8134166,
                337.22626,
                250.61284,
                250.76686,
                501.46893,
                181.5249,
                46.034687,
                1.1643934,
                1.8146939,
                0.84029704,
                2.1608524,
                867.87646,
                237.435,
                237.59702,
                1025.1486,
                177.2652,
                762.4481,
                1.8735278,
                1.8137742,
                3.4653602,
                2.5064802,
            ]
        )
        super().__init__(n_parameters=2, n_observables=48, n_final=4, n_additional_constraints=0, prior_scale=1.0, x_means=X_MEANS, x_stds=X_STDS)


class WBF2DLoader(BaseLHCLoader):
    """
    Features:
    0 pt_j1
    1 deltaphi_aa
    """

    def __init__(self):
        X_MEANS = np.array([264.52158, -0.00532419])
        X_STDS = np.array([171.67308, 2.5071378])
        super().__init__(n_parameters=2, n_observables=2, n_final=4, n_additional_constraints=0, prior_scale=1.0, x_means=X_MEANS, x_stds=X_STDS)
