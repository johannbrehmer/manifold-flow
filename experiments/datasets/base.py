import numpy as np
import os
import logging
from .utils import download_file_from_google_drive, NumpyDataset

logger = logging.getLogger(__name__)


class IntractableLikelihoodError(Exception):
    pass


class DatasetNotAvailableError(Exception):
    pass


class BaseSimulator:
    def __init__(self):
        self.gdrive_file_ids = None

    def is_image(self):
        raise NotImplementedError

    def data_dim(self):
        raise NotImplementedError

    def full_data_dim(self):
        return np.prod(self.data_dim())

    def latent_dim(self):
        raise NotImplementedError

    def parameter_dim(self):
        raise NotImplementedError

    def log_density(self, x, parameters=None):
        raise IntractableLikelihoodError

    def load_dataset(self, train, dataset_dir, numpy=False, limit_samplesize=None, true_param_id=0, joint_score=False, ood=False, run=0):
        if joint_score is not None:
            raise NotImplementedError("SCANDAL training not implemented for this dataset")
        if ood and not os.path.exists("{}/x_ood.npy".format(dataset_dir)):
            raise DatasetNotAvailableError

        # Download missing data
        self._download(dataset_dir)

        tag = "train" if train else "ood" if ood else "paramscan" if paramscan else "test"
        param_label = true_param_id if not train and true_param_id > 0 else ""
        run_label = "_run{}".format(run) if run > 0 else ""

        x = np.load("{}/x_{}{}{}.npy".format(dataset_dir, tag, param_label, run_label))
        if self.parameter_dim() is not None:
            params = np.load("{}/theta_{}{}{}.npy".format(dataset_dir, tag, param_label, run_label))
        else:
            params = np.ones(x.shape[0])

        if limit_samplesize is not None:
            logger.info("Only using %s of %s available samples", limit_samplesize, x.shape[0])
            x = x[:limit_samplesize]
            params = params[:limit_samplesize]

        if numpy:
            return x, params
        else:
            return NumpyDataset(x, params)

    def sample(self, n, parameters=None):
        raise NotImplementedError

    def sample_with_noise(self, n, noise, parameters=None):
        x = self.sample(n, parameters)
        x = x + np.random.normal(loc=0.0, scale=noise, size=(n, self.data_dim()))
        return x

    def sample_ood(self, n, parameters=None):
        raise NotImplementedError

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def default_parameters(self, true_param_id=0):
        return np.zeros(self.parameter_dim())

    def eval_parameter_grid(self, resolution=11):
        if self.parameter_dim() is None or self.parameter_dim() < 1:
            raise NotImplementedError

        each = np.linspace(-1.0, 1.0, resolution)
        each_grid = np.meshgrid(*[each for _ in range(self.parameter_dim())], indexing="ij")
        each_grid = [x.flatten() for x in each_grid]
        grid = np.vstack(each_grid).T
        return grid

    def sample_from_prior(self, n):
        raise NotImplementedError

    def evaluate_log_prior(self, parameters):
        raise NotImplementedError

    def _download(self, dataset_dir):
        if self.gdrive_file_ids is None:
            return

        os.makedirs(dataset_dir, exist_ok=True)

        for tag, file_id in self.gdrive_file_ids.items():
            filename = "{}/{}.npy".format(dataset_dir, tag)
            if not os.path.isfile(filename):
                logger.info("Downloading {}.npy".format(tag))
                download_file_from_google_drive(file_id, filename)
