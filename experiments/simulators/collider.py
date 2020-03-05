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
        latent_dim = 4 * n_final  # We don't assume an on-shell condition for the final states (e.g. for jets)
        latent_dim -= n_additional_constraints  # Additional constraints, for instance from intermediate narrow resonances, and from final-state on-shell conditions
        # # or if you want to impose energy-momentum conservation
        return latent_dim

    def _preprocess(self, x, inverse=False):
        if self._x_means is not None and self._x_stds is not None:
            if inverse:
                x *= self._x_stds
                x += self._x_means
            else:
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
        super().__init__(n_parameters=2, n_observables=48, n_final=4, n_additional_constraints=2, prior_scale=1.0, x_means=X_MEANS, x_stds=X_STDS)

    def _on_shell_discrepancy(self, x_raw, id_e, id_px, id_py, id_pz, m=0.0):
        e_expected = (x_raw[:, id_px] ** 2 + x_raw[:, id_py] ** 2 + x_raw[:, id_pz] ** 2 + m ** 2) ** 0.5
        return np.mean(np.abs(x_raw[:, id_e] - e_expected))

    def _conservation_discrepancy(self, x_raw, idx):
        px = np.sum(x_raw[:, idx], axis=1)
        return np.mean(np.abs(px))

    def _daughter_discrepancy(self, x_raw, id, id_daughter1, id_daughter2):
        return np.mean(np.abs(x_raw[:, id] - x_raw[:, id_daughter1] - x_raw[:, id_daughter2]))

    def _delta_discrepancy(self, x_raw, id, id_daughter1, id_daughter2):
        return np.mean(np.abs(np.abs(x_raw[:, id]) - np.abs(x_raw[:, id_daughter1] - x_raw[:, id_daughter2])))

    def _pt_discrepancy(self, x_raw, id_pt, id_px, id_py):
        pt_expected = (x_raw[:, id_px] ** 2 + x_raw[:, id_py] ** 2) ** 0.5
        return np.mean(np.abs(x_raw[:, id_pt] - pt_expected))

    def _phi_discrepancy(self, x_raw, id_phi, id_px, id_py):
        phi_expected = np.arctan2(x_raw[:, id_py], x_raw[:, id_px])
        return np.mean(np.minimum(
            np.abs(x_raw[:, id_phi] - phi_expected),
            np.abs(2.0 * np.pi + x_raw[:, id_phi] - phi_expected),
            np.abs(-2.0 * np.pi + x_raw[:, id_phi] - phi_expected),
        ))

    def _eta_discrepancy(self, x_raw, id_eta, id_e, id_px, id_py, id_pz):
        costheta = x_raw[:, id_pz] / (x_raw[:, id_px] ** 2 + x_raw[:, id_py] ** 2 + x_raw[:, id_pz]**2) ** 0.5
        eta_expected = -0.5 * np.log((1 - costheta) / (1 + costheta))
        return np.mean(np.abs(x_raw[:, id_eta] - eta_expected))

    def distance_from_manifold(self, x):
        """ Closure test for E-p conservation and on-shell conditions and fixed relations between variables. 34 constraints + 14-dimensional manifold= 48-dimensional data..."""

        # Undo scaling
        x_ = self._preprocess(x, inverse=True)

        d = 0.0

        # pT vs px, py
        d += self._pt_discrepancy(x_, 4, 1, 2)
        d += self._pt_discrepancy(x_, 11, 8, 9)
        d += self._pt_discrepancy(x_, 18, 15, 16)
        d += self._pt_discrepancy(x_, 25, 22, 23)
        d += self._pt_discrepancy(x_, 32, 29, 30)
        d += self._pt_discrepancy(x_, 42, 39, 40)

        # phi vs px, py
        d += self._phi_discrepancy(x_, 6, 1, 2)
        d += self._phi_discrepancy(x_, 13, 8, 9)
        d += self._phi_discrepancy(x_, 20, 15, 16)
        d += self._phi_discrepancy(x_, 27, 22, 23)
        d += self._phi_discrepancy(x_, 35, 29, 30)
        d += self._phi_discrepancy(x_, 45, 39, 40)

        # eta vs E, px, py, pz
        d += self._eta_discrepancy(x_, 5, 0, 1, 2, 3)
        d += self._eta_discrepancy(x_, 12, 7, 8, 9, 10)
        d += self._eta_discrepancy(x_, 19, 14, 15, 16, 17)
        d += self._eta_discrepancy(x_, 26, 21, 22, 23, 24)
        d += self._eta_discrepancy(x_, 34, 28, 29, 30, 31)
        d += self._eta_discrepancy(x_, 44, 38, 39, 40, 41)

        # E vs on-shell and m vs on-shell
        d += self._on_shell_discrepancy(x_, 0, 1, 2, 3)
        d += self._on_shell_discrepancy(x_, 7, 8, 9, 10)
        d += self._on_shell_discrepancy(x_, 28, 29, 30, 31, m=x_[:, 33])
        d += self._on_shell_discrepancy(x_, 38, 39, 40, 41, m=x_[:, 43])

        # sum(pT) vs energy-momentum conservation
        # d += self._conservation_discrepancy(x_, [1, 8, 15, 22])
        # d += self._conservation_discrepancy(x_, [2, 9, 16, 23])

        # reconstructed particles vs daughters
        for add in [0, 1, 2, 3]:
            d += self._daughter_discrepancy(x_, 28 + add, 0 + add, 7 + add)
            d += self._daughter_discrepancy(x_, 38 + add, 14 + add, 21 + add)

        # delta something discrepancies
        d += self._delta_discrepancy(x_, 36, 5, 12)
        d += self._delta_discrepancy(x_, 37, 6, 13)
        d += self._delta_discrepancy(x_, 46, 19, 26)
        d += self._delta_discrepancy(x_, 47, 20, 27)

        return d


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
