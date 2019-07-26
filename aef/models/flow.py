from torch import nn
import logging

from aef.models import vector_transforms, image_transforms
from aef.utils import product
from nsf.nde import distributions


logger = logging.getLogger(__name__)


class Flow(nn.Module):
    def __init__(
        self,
        data_dim,
        mode="vector",
        transform="rq-coupling",
        steps=3,
    ):
        super(Flow, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = data_dim
        self.total_data_dim = product(data_dim)
        self.total_latent_dim = product(self.latent_dim)
        self.mode = mode
        self.latent_distribution = distributions.StandardNormal((self.total_latent_dim,))

        if mode == "vector":
            self.transform = vector_transforms.create_transform(
                data_dim, steps, base_transform_type=transform
            )

        elif mode == "image":
            c, h, w = data_dim
            self.transform = image_transforms.create_transform(
                c, h, w, steps
            )

        self._report_model_parameters()

    def forward(self, x):
        # Encode
        u, log_det = self._encode(x)

        # Decode
        x = self.decode(u)

        # Log prob
        log_prob = self.latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + log_det

        return x, log_prob, u

    def encode(self, x):
        u, _ = self._encode(x)
        return u

    def decode(self, u):
        x, _ = self.transform.inverse(u)
        return x

    def log_prob(self, x):
        # Encode
        u, log_det = self._encode(x)

        # Log prob
        log_prob = self.latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + log_det

        return log_prob

    def sample(self, u=None, n=1):
        if u is None:
            u = self.latent_distribution.sample(n)
        x = self.decode(u)
        return x

    def _encode(self, x):
        u, log_det = self.transform(x)
        return u, log_det

    def _report_model_parameters(self):
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size = all_params * (32 / 8)  # Bytes
        logger.debug(
            "Created autoencoding flow with %.1f M parameters (%.1f M trainable) with an estimated size of %.1f GB",
            all_params / 1e6,
            trainable_params / 1.0e6,
            size / 1.0e9,
        )