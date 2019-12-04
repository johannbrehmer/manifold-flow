import torch
import logging

from manifold_flow.transforms import ProjectionSplit
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
        pie_epsilon=1.0e-3,
        apply_context_to_outer=True,
    ):
        super(ManifoldFlow, self).__init__()

        assert latent_dim < data_dim

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.total_data_dim = product(data_dim)
        self.total_latent_dim = product(latent_dim)
        self.apply_context_to_outer = apply_context_to_outer

        self.manifold_latent_distribution = distributions.StandardNormal((self.total_latent_dim,))
        self.orthogonal_latent_distribution = distributions.RescaledNormal((self.total_data_dim - self.total_latent_dim,), std=pie_epsilon)
        self.projection = ProjectionSplit(self.total_data_dim, self.total_latent_dim)

        self.outer_transform = outer_transform
        if inner_transform is None:
            self.inner_transform = transforms.IdentityTransform()
        else:
            self.inner_transform = inner_transform

        self._report_model_parameters()

    def forward(self, x, mode="mf", context=None):
        """ mode can be "mf" (calculating the exact manifold density based on the full Jacobian), "pie" (calculating the density in x), "slice"
        (calculating the density on x, but projected onto the manifold), or "projection" (calculating no density at all). """

        assert mode in ["mf", "pie", "slice", "projection"]

        # Project to the manifold
        if mode == "slice":
            x = self.project(x, context=context)

        # Encode
        u, h_manifold, h_orthogonal, log_det_inner, log_det_outer, jacobian_outer = self._encode(x, mode, context)

        # Decode
        if not mode == "slice":
            x = self.decode(u, context=context)

        # Log prob
        log_prob = self._log_prob(mode, u, h_orthogonal, log_det_inner, log_det_outer, jacobian_outer)

        return x, log_prob, u

    def encode(self, x, context=None):
        u, _, _, _, _, _ = self._encode(x, mode="projection", context=context)
        return u

    def decode(self, u, u_orthogonal=None, context=None):
        h, _ = self.inner_transform.inverse(u, context=context)
        if u_orthogonal is not None:
            h = self.projection.inverse(h, orthogonal_inputs=u_orthogonal)
        else:
            h = self.projection.inverse(h)
        x, _ = self.outer_transform.inverse(h, context=context if self.apply_context_to_outer else None)
        return x

    def log_prob(self, x, mode="mf", context=None):
        assert mode in ["mf", "pie", "slice", "projection"]

        # Project to the manifold
        if mode == "slice":
            x = self.project(x, context=context)

        # Encode
        u, _, h_orthogonal, log_det_inner, log_det_outer, jacobian_outer = self._encode(x, mode, context)

        # Log prob
        log_prob = self._log_prob(mode, u, h_orthogonal, log_det_inner, log_det_outer, jacobian_outer)

        return log_prob

    def sample(self, u=None, n=1, context=None, sample_orthogonal=False):
        """ Note: this is PIE / MF sampling! Cannot sample from slice of PIE efficiently."""

        if u is None:
            u = self.manifold_latent_distribution.sample(n, context=context)
        u_orthogonal = self.orthogonal_latent_distribution.sample(n, context=context) if sample_orthogonal else None
        x = self.decode(u, u_orthogonal=u_orthogonal)
        return x

    def _encode(self, x, mode, context=None):
        assert mode in ["mf", "pie", "slice", "projection"]

        # Encode
        if mode in ["pie", "slice", "projection"]:
            h, log_det_outer = self.outer_transform(x, context=context if self.apply_context_to_outer else None)
            jacobian_outer = None
        else:
            x.requires_grad = True
            h, jacobian_outer = self.outer_transform(x, full_jacobian=True, context=context if self.apply_context_to_outer else None)
            log_det_outer = None
        h_manifold, h_orthogonal = self.projection(h)
        u, log_det_inner = self.inner_transform(h_manifold, context=context)

        return u, h_manifold, h_orthogonal, log_det_inner, log_det_outer, jacobian_outer

    def _log_prob(self, mode, u, h_orthogonal, log_det_inner, log_det_outer, jacobian_outer):
        assert mode in ["mf", "pie", "slice", "projection"]

        if mode in ["pie", "slice"]:
            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob + self.orthogonal_latent_distribution._log_prob(h_orthogonal, context=None)
            log_prob = log_prob + log_det_outer + log_det_inner
        elif mode == "mf":
            # The Jacobian calculated so far is du / dx, need to invert this to get to dx / du
            jacobian_outer = torch.inverse(jacobian_outer)
            # Next, have to restrict the u space to the manifold direction
            jacobian_outer = jacobian_outer[:, :, : self.latent_dim]
            # And finally calculate log det (J^T J)
            jtj = torch.bmm(torch.transpose(jacobian_outer, -2, -1), jacobian_outer)
            log_det_outer = -0.5 * torch.slogdet(jtj)[1]

            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob + log_det_outer + log_det_inner
        else:
            log_prob = None

        return log_prob
