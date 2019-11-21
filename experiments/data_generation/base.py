class BaseSimulator:
    def data_dim(self):
        raise NotImplementedError

    def latent_dim(self):
        raise NotImplementedError

    def log_density(self, x):
        raise NotImplementedError

    def sample(self, n):
        raise NotImplementedError
