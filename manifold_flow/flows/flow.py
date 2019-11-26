from torch import nn
import logging

from manifold_flow.utils import vector_transforms, image_transforms
from manifold_flow.utils.various import product
from manifold_flow import distributions


logger = logging.getLogger(__name__)


class Flow(nn.Module):
    def __init__(self, data_dim, transform="affine-autoregressive", steps=3, context_features=None):
        super(Flow, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = data_dim
        self.total_data_dim = product(data_dim)
        self.total_latent_dim = product(self.latent_dim)

        self.latent_distribution = distributions.StandardNormal((self.total_latent_dim,))

        if isinstance(self.data_dim, int):
            if isinstance(transform, str):
                logger.debug("Creating default outer transform for scalar data with base type %s", transform)
                self.transform = vector_transforms.create_transform(data_dim, steps, base_transform_type=transform, context_features=context_features)
            else:
                self.transform = transform
        else:
            c, h, w = data_dim
            if isinstance(transform, str):
                logger.debug("Creating default outer transform for image data")
                assert context_features is None
                self.outer_transform = image_transforms.create_transform(c, h, w, steps)
            else:
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

    def _report_model_parameters(self):
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size = all_params * (32 / 8)  # Bytes
        logger.debug("Created standard flow with %.1f M parameters (%.1f M trainable) with an estimated size of %.1f GB", all_params / 1e6, trainable_params / 1.0e6, size / 1.0e9)
