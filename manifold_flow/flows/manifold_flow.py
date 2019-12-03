import torch
import logging

from manifold_flow.transforms import Projection
from manifold_flow.utils.various import product
from manifold_flow import distributions, transforms
from manifold_flow.flows import BaseFlow

logger = logging.getLogger(__name__)


class ManifoldFlow(BaseFlow):
    def __init__(
        self,
        data_dim,
        latent_dim,
        outer_transform,
        inner_transform=None,
        apply_context_to_outer=True,
    ):
        super(ManifoldFlow, self).__init__()

        assert latent_dim < data_dim

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.total_data_dim = product(data_dim)
        self.total_latent_dim = product(latent_dim)
        self.apply_context_to_outer = apply_context_to_outer

        self.latent_distribution = distributions.StandardNormal((self.total_latent_dim,))
        self.projection = Projection(self.total_data_dim, self.total_latent_dim)
        self.outer_transform = outer_transform
        if inner_transform is None:
            self.inner_transform = transforms.IdentityTransform()
        else:
            self.inner_transform = inner_transform

        self._report_model_parameters()

    def forward(self, x, context=None):
        # Encode
        u, h, log_det_inner, jacobian_outer = self._encode(x, context=context)

        # Decode
        x = self.decode(u, context=context)

        # Log prob
        log_prob = self._log_prob(u, log_det_inner, jacobian_outer)

        return x, log_prob, u

    def encode(self, x, context=None):
        u, _, _, _ = self._encode(x, context=context, calculate_jacobian=False)
        return u

    def decode(self, u, context=None):
        h, _ = self.inner_transform.inverse(u, context=context)
        h = self.projection.inverse(h)
        x, _ = self.outer_transform.inverse(h, context=context if self.apply_context_to_outer else None)
        return x

    def log_prob(self, x, context=None):
        # Encode
        u, _, log_det_inner, jacobian_outer = self._encode(x, context=context)

        # Log prob
        log_prob = self._log_prob(u, log_det_inner, jacobian_outer)

        return log_prob

    def sample(self, u=None, n=1, context=None):
        if u is None:
            u = self.latent_distribution.sample(n)
        x = self.decode(u, context=context)
        return x

    def _encode(self, x, calculate_jacobian=True, context=None):
        if calculate_jacobian:
            x.requires_grad = True
        h, jacobian_outer = self.outer_transform(x, full_jacobian=calculate_jacobian, context=context if self.apply_context_to_outer else None)
        h = self.projection(h)
        u, log_det_inner = self.inner_transform(h, context=context)
        if calculate_jacobian:
            return u, h, log_det_inner, jacobian_outer
        else:
            return u, h, None, None

    def _log_prob(self, u, log_det_inner, jacobian_outer):
        jacobian_outer = jacobian_outer[:, :, : self.latent_dim]
        jtj = torch.bmm(torch.transpose(jacobian_outer, -2, -1), jacobian_outer)
        log_det_outer = -0.5 * torch.slogdet(jtj)[1]

        log_prob = self.latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + log_det_outer + log_det_inner

        return log_prob
