import numpy as np


class BaseSimulator:
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
        raise NotImplementedError

    def load_dataset(self, train, dataset_dir):
        raise NotImplementedError

    def sample(self, n, parameters=None):
        raise NotImplementedError

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def default_parameters(self):
        raise NotImplementedError

    def sample_from_prior(self, n):
        raise NotImplementedError

    def evaluate_log_prior(self, parameters):
        raise NotImplementedError
