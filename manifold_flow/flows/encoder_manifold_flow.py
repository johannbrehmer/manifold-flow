import torch
import logging

from manifold_flow.transforms import ProjectionSplit
from manifold_flow.utils.various import product
from manifold_flow import distributions, transforms
from manifold_flow.flows import BaseFlow

logger = logging.getLogger(__name__)


class EncoderManifoldFlow(BaseFlow):
    def __init__(self, data_dim, latent_dim, encoder, outer_transform, inner_transform=None, pie_epsilon=1.0e-2, apply_context_to_outer=True):
        super(EncoderManifoldFlow, self).__init__()

        assert latent_dim < data_dim

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.total_data_dim = product(data_dim)
        self.total_latent_dim = product(latent_dim)
        self.apply_context_to_outer = apply_context_to_outer

        self.manifold_latent_distribution = distributions.StandardNormal((self.total_latent_dim,))
        self.orthogonal_latent_distribution = distributions.RescaledNormal(
            (self.total_data_dim - self.total_latent_dim,), std=pie_epsilon, clip=5.0 * pie_epsilon
        )
        self.projection = ProjectionSplit(self.total_data_dim, self.total_latent_dim)

        self.encoder = encoder
        self.outer_transform = outer_transform
        if inner_transform is None:
            self.inner_transform = transforms.IdentityTransform()
        else:
            self.inner_transform = inner_transform

        self._report_model_parameters()

    def forward(self, x, mode="mf", context=None):
        """ mode can be "mf" (calculating the exact manifold density based on the full Jacobian), "pie" (calculating the density in x), "slice"
        (calculating the density on x, but projected onto the manifold), or "projection" (calculating no density at all). """

        assert mode in ["mf", "projection"]

        if mode == "mf" and not x.requires_grad:
            x.requires_grad = True

        # Encode
        u, h_manifold, log_det_inner = self._encode(x, context)

        # Decode
        x_reco, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer, h_manifold_reco = self._decode(u, mode=mode, context=context)

        # Log prob
        log_prob = self._log_prob(mode, u, log_det_inner, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer)

        return x_reco, log_prob, u

    def encode(self, x, context=None):
        u, _, _, _ = self._encode(x, context=context)
        return u

    def decode(self, u, u_orthogonal=None, context=None):
        x, _, _, _, _ = self._decode(u, mode="projection", u_orthogonal=u_orthogonal, context=context)
        return x

    def log_prob(self, x, mode="mf", context=None):
        return self.forward(x, mode, context)[1]

    def sample(self, u=None, n=1, context=None, sample_orthogonal=False):
        """ Note: this is PIE / MF sampling! Cannot sample from slice of PIE efficiently."""

        if u is None:
            u = self.manifold_latent_distribution.sample(n, context=None)
        u_orthogonal = self.orthogonal_latent_distribution.sample(n, context=None) if sample_orthogonal else None
        x = self.decode(u, u_orthogonal=u_orthogonal, context=context)
        return x

    def _encode(self, x, context=None):
        # Encode
        h_manifold = self.encoder(x, context=context if self.apply_context_to_outer else None)
        u, log_det_inner = self.inner_transform(h_manifold, full_jacobian=False, context=context)

        return u, h_manifold, log_det_inner

    def _decode(self, u, mode, u_orthogonal=None, context=None):
        if mode == "mf" and not u.requires_grad:
            u.requires_grad = True

        h, inv_log_det_inner = self.inner_transform.inverse(u, full_jacobian=False, context=context)

        if u_orthogonal is not None:
            h = self.projection.inverse(h, orthogonal_inputs=u_orthogonal)
        else:
            h = self.projection.inverse(h)

        if mode in ["pie", "slice", "projection"]:
            x, inv_log_det_outer = self.outer_transform.inverse(h, full_jacobian=False, context=context if self.apply_context_to_outer else None)
            inv_jacobian_outer = None
        else:
            x, inv_jacobian_outer = self.outer_transform.inverse(h, full_jacobian=True, context=context if self.apply_context_to_outer else None)
            inv_log_det_outer = None

        if torch.isnan(x).any():
            logger.warning("Reconstructed x contains NaN")
            filter = torch.isnan(x).any(dim=-1).flatten()
            logger.warning("  u: %s", u[filter])
            logger.warning("  h: %s", h[filter])
            logger.warning("  x: %s", x[filter])

        return x, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer, h

    def _log_prob(self, mode, u, log_det_inner, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer):
        if mode == "mf":
            # inv_jacobian_outer is dx / du, but still need to restrict this to the manifold latents
            inv_jacobian_outer = inv_jacobian_outer[:, :, : self.latent_dim]
            # And finally calculate log det (J^T J)
            jtj = torch.bmm(torch.transpose(inv_jacobian_outer, -2, -1), inv_jacobian_outer)

            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob - 0.5 * torch.slogdet(jtj)[1] - inv_log_det_inner

            if torch.isnan(log_prob).any():
                logger.warning("MF log likelihood contains NaNs")
                filter = torch.isnan(log_prob).flatten()
                logger.warning("  u:             %s", u[filter])
                logger.warning("  base density:  %s", self.manifold_latent_distribution._log_prob(u, context=None)[filter])
                logger.warning("  Jacobian:      %s", inv_jacobian_outer[filter])
                logger.warning("  JTJ:           %s", jtj[filter])
                logger.warning("  log det outer: %s", torch.slogdet(jtj)[1][filter])
                logger.warning("  log det inner: %s", inv_log_det_inner[filter])
                logger.warning("  total:         %s", log_prob[filter])

        else:
            log_prob = None

        return log_prob
