import numpy as np


class IntractableLikelihoodError(Exception):
    pass


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
        raise IntractableLikelihoodError

    def load_dataset(self, train, dataset_dir, limit_samplesize=None):
        raise NotImplementedError

    def sample(self, n, parameters=None):
        raise NotImplementedError

    def sample_with_noise(self, n, noise, parameters=None):
        x = self.sample(n, parameters)
        x = x + np.random.normal(loc=0., scale=noise, size=(n, self.data_dim()))
        return x

    def sample_ood(self, n, parameters=None):
        raise NotImplementedError

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def default_parameters(self):
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
