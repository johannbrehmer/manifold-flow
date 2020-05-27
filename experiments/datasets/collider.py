import logging
import numpy as np
from scipy.stats import norm

from .utils import NumpyDataset
from .base import BaseSimulator

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

    def load_dataset(self, train, dataset_dir, numpy=False, limit_samplesize=None, true_param_id=0, joint_score=False, ood=False, paramscan=False, run=0):
        if ood or paramscan:
            raise NotImplementedError()

        # Download missing data
        self._download(dataset_dir)

        # Load numpy arrays
        x = np.load("{}/x_{}{}.npy".format(dataset_dir, "train" if train else "test", true_param_id if not train and true_param_id > 0 else ""))
        params = np.load("{}/theta_{}{}.npy".format(dataset_dir, "train" if train else "test", true_param_id if not train and true_param_id > 0 else ""))
        if joint_score:
            scores = np.load("{}/t_xz_{}{}.npy".format(dataset_dir, "train" if train else "test", true_param_id if not train and true_param_id > 0 else ""))
        else:
            scores = None

        # OPtionally limit sample size
        if limit_samplesize is not None:
            logger.info("Only using %s of %s available samples", limit_samplesize, x.shape[0])
            x = x[:limit_samplesize]
            params = params[:limit_samplesize]
            if joint_score:
                scores = scores[:limit_samplesize]

        # Debug output
        logger.debug("lhc features before preprocessing:")
        for i in range(x.shape[1]):
            logger.debug("  %s: range %s ... %s, mean %s, std %s", i, np.min(x[:, i]), np.max(x[:, i]), np.mean(x[:, i]), np.std(x[:, i]))

        # Preprocess to zero mean and unit variance
        x = self._preprocess(x)

        # Debug output
        logger.debug("lhc features after preprocessing:")
        for i in range(x.shape[1]):
            logger.debug("  %s: range %s ... %s, mean %s, std %s", i, np.min(x[:, i]), np.max(x[:, i]), np.mean(x[:, i]), np.std(x[:, i]))
        logger.debug("lhc parameters:")
        for i in range(params.shape[1]):
            logger.debug("  %s: range %s ... %s, mean %s, std %s", i, np.min(params[:, i]), np.max(params[:, i]), np.mean(params[:, i]), np.std(params[:, i]))

        if numpy and joint_score:
            return x, params, scores
        elif numpy:
            return x, params
        elif joint_score:
            return NumpyDataset(x, params, scores)
        else:
            return NumpyDataset(x, params)

    def sample(self, n, parameters=None):
        raise NotImplementedError

    def sample_ood(self, n, parameters=None):
        raise NotImplementedError

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def sample_from_prior(self, n):
        return np.random.normal(loc=np.zeros((n, self._parameter_dim)), scale=self._prior_scale * np.ones((n, self._parameter_dim)), size=(n, self._parameter_dim))

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

        self.CLOSURE_TEST_WEIGHTS = np.array(
            [
                0.005733124397251814,
                0.019373627498998545,
                0.005601847357892751,
                0.013246297554593486,
                0.0051492718104518736,
                0.005285928756022291,
                0.5434573654744188,
                0.5495540233428311,
                0.548574528639762,
                0.5499645858762766,
                0.5443319900983565,
                0.5477470269075005,
                0.751120437331326,
                0.6837450347638595,
                0.479495732489328,
                0.35411560084702814,
                0.6944403914297372,
                0.4581065620636643,
                0.00338315127519218,
                0.008990040230563823,
                0.0029443731703010264,
                0.0011509776322986742,
                0.0027066953197164415,
                0.0010181997312659812,
                0.0037623941358973473,
                0.003735274410662914,
                0.0037745140586529515,
                0.0037235471577507923,
                0.0018995239678828314,
                0.0007850227402417481,
                1.1083823994444644,
                0.6246407123110107,
                0.47202092213106334,
                0.5863775777004946,
            ]
        )

        self.CLOSURE_LABELS = ["pt"] * 6 + ["phi"] * 6 + ["eta"] * 6 + ["on-shell"] * 4 + ["decay"] * 8 + ["delta"] * 2

    def default_parameters(self, true_param_id=0):
        if true_param_id == 1:
            return np.array([0.5, 0.0])
        elif true_param_id == 2:
            return np.array([-1.0, -1.0])
        else:
            return np.zeros(self._parameter_dim)

    def _on_shell_discrepancy(self, x_raw, id_e, id_px, id_py, id_pz, m=0.0):
        e_expected = (x_raw[:, id_px] ** 2 + x_raw[:, id_py] ** 2 + x_raw[:, id_pz] ** 2 + m ** 2) ** 0.5
        return np.abs(x_raw[:, id_e] - e_expected)

    def _conservation_discrepancy(self, x_raw, idx):
        px = np.sum(x_raw[:, idx], axis=1)
        return np.abs(px)

    def _daughter_discrepancy(self, x_raw, id, id_daughter1, id_daughter2):
        return np.abs(x_raw[:, id] - x_raw[:, id_daughter1] - x_raw[:, id_daughter2])

    def _delta_discrepancy(self, x_raw, id, id_daughter1, id_daughter2):
        return np.abs(np.abs(x_raw[:, id]) - np.abs(x_raw[:, id_daughter1] - x_raw[:, id_daughter2]))

    def _pt_discrepancy(self, x_raw, id_pt, id_px, id_py):
        pt_expected = (x_raw[:, id_px] ** 2 + x_raw[:, id_py] ** 2) ** 0.5
        return np.abs(x_raw[:, id_pt] - pt_expected)

    def _phi_discrepancy(self, x_raw, id_phi, id_px, id_py):
        phi_expected = np.arctan2(x_raw[:, id_py], x_raw[:, id_px])
        return np.minimum(np.abs(x_raw[:, id_phi] - phi_expected), np.abs(2.0 * np.pi + x_raw[:, id_phi] - phi_expected), np.abs(-2.0 * np.pi + x_raw[:, id_phi] - phi_expected),)

    def _eta_discrepancy(self, x_raw, id_eta, id_e, id_px, id_py, id_pz):
        costheta = x_raw[:, id_pz] / (x_raw[:, id_px] ** 2 + x_raw[:, id_py] ** 2 + x_raw[:, id_pz] ** 2) ** 0.5
        eta_expected = -0.5 * np.log((1 - costheta) / (1 + costheta))
        return np.abs(x_raw[:, id_eta] - eta_expected)

    def _closure_tests(self, x):
        """ Closure test. 34 constraints + 14-dimensional manifold = 48-dimensional data..."""

        # Undo scaling
        x_ = self._preprocess(x, inverse=True)

        # Begin closure tests
        closure_tests = []

        # pT vs px, py
        closure_tests.append(self._pt_discrepancy(x_, 4, 1, 2))
        closure_tests.append(self._pt_discrepancy(x_, 11, 8, 9))
        closure_tests.append(self._pt_discrepancy(x_, 18, 15, 16))
        closure_tests.append(self._pt_discrepancy(x_, 25, 22, 23))
        closure_tests.append(self._pt_discrepancy(x_, 32, 29, 30))
        closure_tests.append(self._pt_discrepancy(x_, 42, 39, 40))

        # phi vs px, py
        closure_tests.append(self._phi_discrepancy(x_, 6, 1, 2))
        closure_tests.append(self._phi_discrepancy(x_, 13, 8, 9))
        closure_tests.append(self._phi_discrepancy(x_, 20, 15, 16))
        closure_tests.append(self._phi_discrepancy(x_, 27, 22, 23))
        closure_tests.append(self._phi_discrepancy(x_, 35, 29, 30))
        closure_tests.append(self._phi_discrepancy(x_, 45, 39, 40))

        # eta vs E, px, py, pz
        closure_tests.append(self._eta_discrepancy(x_, 5, 0, 1, 2, 3))
        closure_tests.append(self._eta_discrepancy(x_, 12, 7, 8, 9, 10))
        closure_tests.append(self._eta_discrepancy(x_, 19, 14, 15, 16, 17))
        closure_tests.append(self._eta_discrepancy(x_, 26, 21, 22, 23, 24))
        closure_tests.append(self._eta_discrepancy(x_, 34, 28, 29, 30, 31))
        closure_tests.append(self._eta_discrepancy(x_, 44, 38, 39, 40, 41))

        # E vs on-shell and m vs on-shell
        closure_tests.append(self._on_shell_discrepancy(x_, 0, 1, 2, 3))
        closure_tests.append(self._on_shell_discrepancy(x_, 7, 8, 9, 10))
        closure_tests.append(self._on_shell_discrepancy(x_, 28, 29, 30, 31, m=x_[:, 33]))
        closure_tests.append(self._on_shell_discrepancy(x_, 38, 39, 40, 41, m=x_[:, 43]))

        # sum(pT) vs energy-momentum conservation
        # closure_tests.append( self._conservation_discrepancy(x_, [1, 8, 15, 22]))
        # closure_tests.append( self._conservation_discrepancy(x_, [2, 9, 16, 23]))

        # reconstructed particles vs daughters
        for add in [0, 1, 2, 3]:
            closure_tests.append(self._daughter_discrepancy(x_, 28 + add, 0 + add, 7 + add))
            closure_tests.append(self._daughter_discrepancy(x_, 38 + add, 14 + add, 21 + add))

        # delta something discrepancies
        closure_tests.append(self._delta_discrepancy(x_, 36, 5, 12))
        closure_tests.append(self._delta_discrepancy(x_, 37, 6, 13))
        closure_tests.append(self._delta_discrepancy(x_, 46, 19, 26))
        closure_tests.append(self._delta_discrepancy(x_, 47, 20, 27))

        closure_tests = np.asarray(closure_tests)
        return closure_tests

    def distance_from_manifold(self, x):
        weighted_closure_tests = self.CLOSURE_TEST_WEIGHTS[:, np.newaxis] * self._closure_tests(x)
        logger.info("Mean closure test results (before clipping and averaging):")
        for closure, label in zip(np.mean(weighted_closure_tests, axis=1), self.CLOSURE_LABELS):
            logger.info("  %5.3f - %s", closure, label)

        weighted_closure_tests = np.mean(np.clip(weighted_closure_tests, 0.0, 10.0), axis=0)
        logger.info("Mean closure test result (after clipping and averaging): %s", np.mean(weighted_closure_tests))
        return weighted_closure_tests


class WBF40DLoader(BaseLHCLoader):
    """
    Features:
    0 e_a1
    1 px_a1
    2 py_a1
    3 pz_a1
    4 pt_a1
    5 eta_a1
    6 e_a2
    7 px_a2
    8 py_a2
    9 pz_a2
    10 pt_a2
    11 eta_a2
    12 e_j1
    13 px_j1
    14 py_j1
    15 pz_j1
    16 pt_j1
    17 eta_j1
    18 e_j2
    19 px_j2
    20 py_j2
    21 pz_j2
    22 pt_j2
    23 eta_j2
    24 e_aa
    25 px_aa
    26 py_aa
    27 pz_aa
    28 pt_aa
    29 m_aa
    30 eta_aa
    31 deltaeta_aa
    32 e_jj
    33 px_jj
    34 py_jj
    35 pz_jj
    36 pt_jj
    37 m_jj
    38 eta_jj
    39 deltaeta_jj
    """

    def __init__(self):
        X_MEANS = np.array(
            [
                399.37213,
                -0.43740344,
                0.6336686,
                -0.6348415,
                249.49158,
                -0.003057861,
                131.7912,
                -0.05828116,
                0.24816507,
                -0.7630951,
                76.36908,
                -0.005826209,
                757.37,
                -0.10769359,
                -0.6086537,
                -6.178487,
                264.52158,
                -0.008442603,
                574.55524,
                0.07781012,
                -0.09404507,
                2.8334167,
                104.1036,
                0.0012078341,
                531.1633,
                -0.49568462,
                0.8818339,
                -1.3979366,
                304.6138,
                127.9752,
                -0.004153981,
                0.0027683484,
                1331.9247,
                -0.029883433,
                -0.7026987,
                -3.3450694,
                285.39395,
                884.056,
                -0.008315955,
                -0.009650437,
            ]
        )
        X_STDS = np.array(
            [
                299.35312,
                211.85915,
                212.03178,
                399.08353,
                166.1214,
                1.0561188,
                111.997505,
                64.16174,
                64.08385,
                147.26929,
                48.90089,
                1.1154486,
                634.2648,
                222.84987,
                223.1163,
                935.7079,
                171.67308,
                1.7555672,
                597.842,
                90.13577,
                90.237915,
                819.1639,
                73.68713,
                2.3172193,
                337.30893,
                250.67441,
                250.83035,
                501.64795,
                181.56274,
                46.119442,
                1.164671,
                0.84049183,
                868.1088,
                237.49751,
                237.65948,
                1025.469,
                177.30736,
                762.6249,
                1.8738469,
                3.466002,
            ]
        )
        super().__init__(n_parameters=2, n_observables=40, n_final=4, n_additional_constraints=2, prior_scale=1.0, x_means=X_MEANS, x_stds=X_STDS)

        self.CLOSURE_TEST_WEIGHTS = np.array(
            [
                0.005733124397251814,
                0.019373627498998545,
                0.005601847357892751,
                0.013246297554593486,
                0.0051492718104518736,
                0.005285928756022291,
                0.751120437331326,
                0.6837450347638595,
                0.479495732489328,
                0.35411560084702814,
                0.6944403914297372,
                0.4581065620636643,
                0.00338315127519218,
                0.008990040230563823,
                0.0029443731703010264,
                0.0011509776322986742,
                0.0027066953197164415,
                0.0010181997312659812,
                0.0037623941358973473,
                0.003735274410662914,
                0.0037745140586529515,
                0.0037235471577507923,
                0.0018995239678828314,
                0.0007850227402417481,
                1.1083823994444644,
                0.47202092213106334,
            ]
        )
        self.CLOSURE_LABELS = ["pt"] * 6 + ["eta"] * 6 + ["on-shell"] * 4 + ["decay"] * 8 + ["delta"] * 2

        self.gdrive_file_ids = {
            "x_train": "1VuG3HTtJHzzQi5KcltMUmxdoADAf34BO",
            "x_test": "1J6lcVmyFYbRPx9R2GHfoQKKDtNbaD2sD",
            "x_test1": "1aRtfaBrOP_XfYCwUaNHPlbDOmGjdbQn3",
            "x_test2": "1X7SRFjIW2sv8gagyfyFs_iUmPG8mbrIe",
            "t_xz_train": "190_Hdu2DLB4k8hKTnxrmmj5YgKt_whMA",
            "theta_train": "1VZljWA63wMOXKAWvGMdMRdDE_TLmAMyV",
            "theta_test": "1O83YYSAbnkz2tESSKEhSXykspDLHHc6E",
            "theta_test1": "100GEgVnIX7bEocTR_0Hyu-8xRDKDLjym",
            "theta_test2": "1vFJYdVzG22ARPuUw7hn-KBF05PRiFDst",
        }

    def default_parameters(self, true_param_id=0):
        if true_param_id == 1:
            return np.array([0.5, 0.0])
        elif true_param_id == 2:
            return np.array([-1.0, -1.0])
        else:
            return np.zeros(self._parameter_dim)

    def _on_shell_discrepancy(self, x_raw, id_e, id_px, id_py, id_pz, m=0.0):
        e_expected = (x_raw[:, id_px] ** 2 + x_raw[:, id_py] ** 2 + x_raw[:, id_pz] ** 2 + m ** 2) ** 0.5
        return np.abs(x_raw[:, id_e] - e_expected)

    def _conservation_discrepancy(self, x_raw, idx):
        px = np.sum(x_raw[:, idx], axis=1)
        return np.abs(px)

    def _daughter_discrepancy(self, x_raw, id, id_daughter1, id_daughter2):
        return np.abs(x_raw[:, id] - x_raw[:, id_daughter1] - x_raw[:, id_daughter2])

    def _delta_discrepancy(self, x_raw, id, id_daughter1, id_daughter2):
        return np.abs(np.abs(x_raw[:, id]) - np.abs(x_raw[:, id_daughter1] - x_raw[:, id_daughter2]))

    def _pt_discrepancy(self, x_raw, id_pt, id_px, id_py):
        pt_expected = (x_raw[:, id_px] ** 2 + x_raw[:, id_py] ** 2) ** 0.5
        return np.abs(x_raw[:, id_pt] - pt_expected)

    def _phi_discrepancy(self, x_raw, id_phi, id_px, id_py):
        phi_expected = np.arctan2(x_raw[:, id_py], x_raw[:, id_px])
        return np.minimum(np.abs(x_raw[:, id_phi] - phi_expected), np.abs(2.0 * np.pi + x_raw[:, id_phi] - phi_expected), np.abs(-2.0 * np.pi + x_raw[:, id_phi] - phi_expected),)

    def _eta_discrepancy(self, x_raw, id_eta, id_e, id_px, id_py, id_pz):
        costheta = x_raw[:, id_pz] / (x_raw[:, id_px] ** 2 + x_raw[:, id_py] ** 2 + x_raw[:, id_pz] ** 2) ** 0.5
        eta_expected = -0.5 * np.log((1 - costheta) / (1 + costheta))
        return np.abs(x_raw[:, id_eta] - eta_expected)

    def _closure_tests(self, x):
        """ Closure test. 34 constraints + 14-dimensional manifold = 48-dimensional data..."""

        # Undo scaling
        x_ = self._preprocess(x, inverse=True)

        # Begin closure tests
        closure_tests = []

        # pT vs px, py
        closure_tests.append(self._pt_discrepancy(x_, 4, 1, 2))
        closure_tests.append(self._pt_discrepancy(x_, 10, 7, 8))
        closure_tests.append(self._pt_discrepancy(x_, 16, 13, 14))
        closure_tests.append(self._pt_discrepancy(x_, 22, 19, 20))
        closure_tests.append(self._pt_discrepancy(x_, 28, 25, 26))
        closure_tests.append(self._pt_discrepancy(x_, 36, 33, 34))

        # eta vs E, px, py, pz
        closure_tests.append(self._eta_discrepancy(x_, 5, 0, 1, 2, 3))
        closure_tests.append(self._eta_discrepancy(x_, 11, 6, 7, 8, 9))
        closure_tests.append(self._eta_discrepancy(x_, 17, 12, 13, 14, 15))
        closure_tests.append(self._eta_discrepancy(x_, 23, 18, 19, 20, 21))
        closure_tests.append(self._eta_discrepancy(x_, 30, 24, 25, 26, 27))
        closure_tests.append(self._eta_discrepancy(x_, 38, 32, 33, 34, 35))

        # E vs on-shell and m vs on-shell
        closure_tests.append(self._on_shell_discrepancy(x_, 0, 1, 2, 3))
        closure_tests.append(self._on_shell_discrepancy(x_, 6, 7, 8, 9))
        closure_tests.append(self._on_shell_discrepancy(x_, 24, 25, 26, 27, m=x_[:, 29]))
        closure_tests.append(self._on_shell_discrepancy(x_, 32, 33, 34, 35, m=x_[:, 37]))

        # reconstructed particles vs daughters
        for add in [0, 1, 2, 3]:
            closure_tests.append(self._daughter_discrepancy(x_, 24 + add, 0 + add, 6 + add))
            closure_tests.append(self._daughter_discrepancy(x_, 32 + add, 12 + add, 18 + add))

        # delta something discrepancies
        closure_tests.append(self._delta_discrepancy(x_, 31, 5, 11))
        closure_tests.append(self._delta_discrepancy(x_, 39, 17, 23))

        closure_tests = np.asarray(closure_tests)
        return closure_tests

    def distance_from_manifold(self, x):
        weighted_closure_tests = self.CLOSURE_TEST_WEIGHTS[:, np.newaxis] * self._closure_tests(x)
        logger.info("Mean closure test results (before clipping and averaging):")
        for closure, label in zip(np.mean(weighted_closure_tests, axis=1), self.CLOSURE_LABELS):
            logger.info("  %5.3f - %s", closure, label)

        weighted_closure_tests = np.mean(np.clip(weighted_closure_tests, 0.0, 10.0), axis=0)
        logger.info("Mean closure test result (after clipping and averaging): %s", np.mean(weighted_closure_tests))
        return weighted_closure_tests


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

    def default_parameters(self, true_param_id=0):
        if true_param_id == 1:
            return np.array([0.5, 0.0])
        elif true_param_id == 2:
            return np.array([-1.0, -1.0])
        else:
            return np.zeros(self._parameter_dim)
