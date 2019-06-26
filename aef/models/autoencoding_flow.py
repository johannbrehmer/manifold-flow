import torch
from torch import nn
import numpy as np

from nsf.nde import distributions, flows, transforms
from aef.models.create_transforms import create_transform


class Projection(transforms.Transform):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        assert input_dim >= output_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, inputs, context=None):
        u = inputs[:, :self.output_dim]
        return u

    def inverse(self, inputs, context=None):
        x = torch.cat((inputs, torch.zeros(inputs.size(0), self.input_dim - self.output_dim)), dim=1)
        return x


class TwoStepAutoencodingFlow(nn.Module):
    def __init__(self, data_dim, latent_dim=10, steps_inner=3, steps_outer=3, n_hidden=100):
        super(TwoStepAutoencodingFlow, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim

        self.outer_transform = create_transform(data_dim, steps_outer)
        self.projection = Projection(data_dim, latent_dim)
        self.inner_transform = create_transform(latent_dim, steps_inner)
        self.latent_distribution = distributions.StandardNormal((latent_dim, ))
        # self.inner_flow = flows.Flow(self.inner_transform, self.latent_distribution)  # Could be a convenient wrapper

    def forward(self, x):
        # Encode
        u, h, log_det_inner, log_det_outer = self._encode(x)

        # Decode
        x = self.decode(u)

        # Log prob
        log_prob = self.latent_distribution._log_prob(u)
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
        h, log_det_outer = self.outer_transform(x)
        h = self.projection(h)
        u, log_det_inner = self.inner_transform(h)

        # Log prob
        log_prob = self.latent_distribution._log_prob(u)
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
