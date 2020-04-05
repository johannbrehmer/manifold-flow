import logging
import torch

from manifold_flow.utils.various import product
from manifold_flow import distributions
from manifold_flow.flows import BaseFlow

logger = logging.getLogger(__name__)


class VariableDimensionManifoldFlow(BaseFlow):
    """ PIE with variable manifold dimensions (learnable epsilons). Work in progress! """

    def __init__(self, data_dim, transform):
        super(VariableDimensionManifoldFlow, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = data_dim
        self.total_data_dim = product(data_dim)
        self.total_latent_dim = product(self.latent_dim)

        self.latent_distribution = distributions.DiagonalNormal((self.total_latent_dim,), initial_std=0.5)
        self.transform = transform

        self._report_model_parameters()

    def forward(self, x, context=None):
        """ Transforms data point to latent space and evaluates likelihood """

        # Encode
        u, log_det = self._encode(x, context=context)

        # Decode
        x = self.decode(u, context=context)

        # Log prob
        log_prob = self.latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + log_det

        return x, log_prob, u

    def encode(self, x, context=None):
        """ Transforms data point to latent space. """

        u, _ = self._encode(x, context=context)
        return u

    def decode(self, u, context=None):
        """ Decodes latent variable to data space."""

        x, _ = self.transform.inverse(u, context=context)
        return x

    def log_prob(self, x, context=None):
        """ Evaluates log likelihood for given data point."""

        # Encode
        u, log_det = self._encode(x, context)

        # Log prob
        log_prob = self.latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + log_det

        return log_prob

    def sample(self, u=None, n=1, context=None):
        """ Generates samples from model. """

        if u is None:
            u = self.latent_distribution.sample(n, context=None)
        x = self.decode(u, context=context)
        return x

    def latent_stds(self):
        """ Returns PIE epsilons """

        return torch.exp(self.latent_distribution.log_stds)

    def calculate_latent_dim(self, threshold=0.5):
        """ Computes learned manifold dimensionality """

        return torch.sum(self.latent_stds() > threshold)

    def latent_regularizer(self, l1=0.0, l2=0.0):
        """ Computes regularization term to force epsilon close to 0 or close to 1 """

        latent_reg = torch.zeros((1,))
        stds = self.latent_stds()
        offset = torch.where(stds > 0.5, stds - 1.0, stds)

        if l1 > 0.0:
            latent_reg = latent_reg + l1 * torch.nn.L1Loss(reduction="sum")(offset, torch.zeros_like(offset))

        if l2 > 0.0:
            latent_reg = latent_reg + 0.5 * l2 * torch.sum(offset ** 2)  # Factor 0.5 to agree with weight_decay

        return latent_reg

    def _encode(self, x, context=None):
        u, log_det = self.transform(x, context=context)
        return u, log_det
