import logging

from manifold_flow.utils.various import product
from manifold_flow import distributions
from manifold_flow.flows import BaseFlow

logger = logging.getLogger(__name__)


class Flow(BaseFlow):
    def __init__(self, data_dim, transform):
        super(Flow, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = data_dim
        self.total_data_dim = product(data_dim)
        self.total_latent_dim = product(self.latent_dim)

        self.latent_distribution = distributions.StandardNormal((self.total_latent_dim,))
        self.transform = transform

        self._report_model_parameters()

    def forward(self, x, context=None):
        # Encode
        u, log_det = self._encode(x, context=context)

        # Decode
        x = self.decode(u, context=context)

        # Log prob
        log_prob = self.latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + log_det

        return x, log_prob, u

    def encode(self, x, context=None):
        u, _ = self._encode(x, context=context)
        return u

    def decode(self, u, context=None):
        x, _ = self.transform.inverse(u, context=context)
        return x

    def log_prob(self, x, context=None):
        # Encode
        u, log_det = self._encode(x, context)

        # Log prob
        log_prob = self.latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + log_det

        return log_prob

    def sample(self, u=None, n=1, context=None):
        if u is None:
            u = self.latent_distribution.sample(n, context=None)
        x = self.decode(u, context=context)
        return x

    def _encode(self, x, context=None):
        u, log_det = self.transform(x, context=context)
        return u, log_det
