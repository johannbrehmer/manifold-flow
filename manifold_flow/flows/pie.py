import torch
from torch import nn
import logging

from manifold_flow.utils.various import product
from experiments.utils import image_transforms, vector_transforms
from manifold_flow import distributions, transforms

logger = logging.getLogger(__name__)


class ProjectionSplit(transforms.Transform):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_dim_total = product(input_dim)
        self.output_dim_total = product(output_dim)
        self.mode_in = "vector" if isinstance(input_dim, int) else "image"
        self.mode_out = "vector" if isinstance(input_dim, int) else "image"

        logger.debug("Set up projection from %s with dimension %s to %s with dimension %s", self.mode_in, self.input_dim, self.mode_out, self.output_dim)

        assert self.input_dim_total >= self.output_dim_total, "Input dimension has to be larger than output dimension"

    def forward(self, inputs, **kwargs):
        if self.mode_in == "vector" and self.mode_out == "vector":
            u = inputs[:, : self.output_dim]
            rest = inputs[:, self.output_dim :]
        elif self.mode_in == "image" and self.mode_out == "vector":
            h = inputs.view(inputs.size(0), -1)
            u = h[:, : self.output_dim]
            rest = h[:, self.output_dim :]
        else:
            raise NotImplementedError("Unsuppoorted projection modes {}, {}".format(self.mode_in, self.mode_out))
        return u, rest

    def inverse(self, inputs, **kwargs):
        orthogonal_inputs = kwargs.get("orthogonal_inputs", torch.zeros(inputs.size(0), self.input_dim_total - self.output_dim))
        if self.mode_in == "vector" and self.mode_out == "vector":
            x = torch.cat((inputs, orthogonal_inputs), dim=1)
        elif self.mode_in == "image" and self.mode_out == "vector":
            c, h, w = self.input_dim
            x = torch.cat((inputs, orthogonal_inputs), dim=1)
            x = x.view(inputs.size(0), c, h, w)
        else:
            raise NotImplementedError("Unsuppoorted projection modes {}, {}".format(self.mode_in, self.mode_out))
        return x


class PIE(nn.Module):
    def __init__(
        self,
        data_dim,
        latent_dim,
        outer_transform,
        inner_transform=None,
        epsilon=1.0e-3,
        apply_context_to_outer=True,
    ):
        super(PIE, self).__init__()

        assert latent_dim < data_dim

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.total_data_dim = product(data_dim)
        self.total_latent_dim = product(latent_dim)
        self.apply_context_to_outer = apply_context_to_outer

        self.manifold_latent_distribution = distributions.StandardNormal((self.total_latent_dim,))
        self.orthogonal_latent_distribution = distributions.RescaledNormal((self.total_data_dim - self.total_latent_dim,), std=epsilon)
        self.projection = ProjectionSplit(self.total_data_dim, self.total_latent_dim)

        self.outer_transform = outer_transform
        if inner_transform is None:
            self.inner_transform = transforms.IdentityTransform()
        else:
            self.inner_transform = inner_transform

        self._report_model_parameters()

    def forward(self, x, context=None):
        # Encode
        u, h, h_orthogonal, log_det_inner, log_det_outer = self._encode(x, context=context)

        # Decode
        x = self.decode(u, context=context)

        # Log prob
        log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + self.orthogonal_latent_distribution._log_prob(h_orthogonal, context=None)
        log_prob = log_prob + log_det_outer + log_det_inner

        return x, log_prob, u

    def encode(self, x, context=None):
        u, _, _, _, _ = self._encode(x, context=context)
        return u

    def decode(self, u, u_orthogonal=None, context=None):
        h, _ = self.inner_transform.inverse(u, context=context)
        if u_orthogonal is not None:
            h = self.projection.inverse(h, orthogonal_inputs=u_orthogonal)
        else:
            h = self.projection.inverse(h)
        x, _ = self.outer_transform.inverse(h, context=context if self.apply_context_to_outer else None)
        return x

    def log_prob(self, x, context=None):
        # Encode
        u, _, h_orthogonal, log_det_inner, log_det_outer = self._encode(x, context=context)

        # Log prob
        log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + self.orthogonal_latent_distribution._log_prob(h_orthogonal, context=None)
        log_prob = log_prob + log_det_outer + log_det_inner

        return log_prob

    def sample(self, u=None, n=1, context=None, sample_orthogonal=False):
        if u is None:
            u = self.manifold_latent_distribution.sample(n, context=context)
        u_orthogonal = self.orthogonal_latent_distribution.sample(n, context=context) if sample_orthogonal else None
        x = self.decode(u, u_orthogonal=u_orthogonal)
        return x

    def _encode(self, x, context=None):
        h, log_det_outer = self.outer_transform(x, context=context if self.apply_context_to_outer else None)
        h_manifold, h_orthogonal = self.projection(h)
        u, log_det_inner = self.inner_transform(h_manifold, context=context)
        return u, h_manifold, h_orthogonal, log_det_inner, log_det_outer

    def _report_model_parameters(self):
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size = all_params * (32 / 8)  # Bytes
        logger.debug("Created PIE with %.1f M parameters (%.1f M trainable) with an estimated size of %.1f GB", all_params / 1e6, trainable_params / 1.0e6, size / 1.0e9)
