import torch
from torch import nn
import logging

from nsf.nde import distributions, transforms
from aef.models import vector_transforms, image_transforms

logger = logging.getLogger(__name__)


class Projection(transforms.Transform):

    # TODO: image case

    def __init__(self, input_dim, output_dim):
        super().__init__()
        assert input_dim >= output_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, inputs, context=None):
        u = inputs[:, : self.output_dim]
        return u

    def inverse(self, inputs, context=None):
        x = torch.cat(
            (inputs, torch.zeros(inputs.size(0), self.input_dim - self.output_dim)),
            dim=1,
        )
        return x


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
        self.mode = mode
        self.latent_distribution = distributions.StandardNormal((self.latent_dim,))

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


class TwoStepAutoencodingFlow(nn.Module):
    def __init__(
        self,
        data_dim,
        latent_dim=10,
        mode="vector",
        inner="rq-coupling",
        outer="rq-coupling",
        steps_inner=3,
        steps_outer=3,
    ):
        super(TwoStepAutoencodingFlow, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.mode = mode
        self.projection = Projection(data_dim, latent_dim)
        self.latent_distribution = distributions.StandardNormal((latent_dim,))

        if mode == "vector":
            self.outer_transform = vector_transforms.create_transform(
                data_dim, steps_outer, base_transform_type=outer
            )
            self.inner_transform = vector_transforms.create_transform(
                latent_dim, steps_inner, base_transform_type=inner
            )

        elif mode == "image":
            c, h, w = data_dim
            self.outer_transform = image_transforms.create_transform(
                c, h, w, steps_outer
            )
            self.inner_transform = vector_transforms.create_transform(
                latent_dim, steps_inner, base_transform_type=inner
            )

        self._report_model_parameters()

    def forward(self, x):
        # Encode
        u, h, log_det_inner, log_det_outer = self._encode(x)

        # Decode
        x = self.decode(u)

        # Log prob
        log_prob = self.latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + log_det_outer + log_det_inner

        return x, log_prob, u

    def encode(self, x):
        u, _, _, _ = self._encode(x)
        return u

    def decode(self, u):
        h, _ = self.inner_transform.inverse(u)
        h = self.projection.inverse(h)
        x, _ = self.outer_transform.inverse(h)
        return x

    def log_prob(self, x):
        # Encode
        u, _, log_det_inner, log_det_outer = self._encode(x)

        # Log prob
        log_prob = self.latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + log_det_outer + log_det_inner

        return log_prob

    def sample(self, u=None, n=1):
        if u is None:
            u = self.latent_distribution.sample(n)
        x = self.decode(u)
        return x

    def _encode(self, x):
        h, log_det_outer = self.outer_transform(x)
        h = self.projection(h)
        u, log_det_inner = self.inner_transform(h)
        return u, h, log_det_inner, log_det_outer

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
