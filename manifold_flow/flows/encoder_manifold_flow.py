import torch
import logging
from torch import nn

from manifold_flow.transforms import ProjectionSplit
from manifold_flow.utils.various import product
from manifold_flow import distributions, transforms
from manifold_flow.flows import BaseFlow

logger = logging.getLogger(__name__)


class EncoderManifoldFlow(BaseFlow):
    """ Manifold-based flow with separate encoder (for MFMFE) """

    def __init__(self, data_dim, latent_dim, encoder, outer_transform, inner_transform=None, pie_epsilon=1.0e-2, apply_context_to_outer=True, clip_pie=False):
        super(EncoderManifoldFlow, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.total_data_dim = product(data_dim)
        self.total_latent_dim = product(latent_dim)
        self.apply_context_to_outer = apply_context_to_outer

        assert self.total_latent_dim < self.total_data_dim

        self.manifold_latent_distribution = distributions.StandardNormal((self.total_latent_dim,))
        self.orthogonal_latent_distribution = distributions.RescaledNormal(
            (self.total_data_dim - self.total_latent_dim,), std=pie_epsilon, clip=None if not clip_pie else clip_pie * pie_epsilon
        )
        self.projection = ProjectionSplit(self.total_data_dim, self.total_latent_dim)

        self.encoder = encoder
        self.outer_transform = outer_transform
        if inner_transform is None:
            self.inner_transform = transforms.IdentityTransform()
        else:
            self.inner_transform = inner_transform

        self._report_model_parameters()

    def forward(self, x, mode="mf", context=None, return_hidden=False):
        """
        Transforms data point to latent space, evaluates likelihood, and transforms it back to data space.

        mode can be "mf" (calculating the exact manifold density based on the full Jacobian), "pie" (calculating the density in x), "slice"
        (calculating the density on x, but projected onto the manifold), or "projection" (calculating no density at all).
        """

        assert mode in ["mf", "projection", "pie", "mf-fixed-manifold"]

        if mode == "mf" and not x.requires_grad:
            x.requires_grad = True

        # Encode
        u, h_manifold, log_det_inner = self._encode(x, context)

        # Decode
        x_reco, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer, h_manifold_reco = self._decode(u, mode=mode, context=context)

        # Log prob
        log_prob = self._log_prob(mode, u, log_det_inner, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer)

        if return_hidden:
            return x_reco, log_prob, u, h_manifold
        return x_reco, log_prob, u

    def encode(self, x, context=None):
        """ Transforms data point to latent space. """

        u, _, _ = self._encode(x, context=context)
        return u

    def decode(self, u, u_orthogonal=None, context=None):
        """ Decodes latent variable to data space."""

        x, _, _, _, _ = self._decode(u, mode="projection", u_orthogonal=u_orthogonal, context=context)
        return x

    def log_prob(self, x, mode="mf", context=None):
        """ Evaluates log likelihood for given data point."""

        return self.forward(x, mode, context)[1]

    def sample(self, u=None, n=1, context=None, sample_orthogonal=False):
        """
        Generates samples from model.

        Note: this is PIE / MF sampling! Cannot sample from slice of PIE efficiently.
        """

        if u is None:
            u = self.manifold_latent_distribution.sample(n, context=None)
        u_orthogonal = self.orthogonal_latent_distribution.sample(n, context=None) if sample_orthogonal else None
        x = self.decode(u, u_orthogonal=u_orthogonal, context=context)
        return x

    def _encode(self, x, context=None):
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

        if mode in ["pie", "slice", "projection", "mf-fixed-manifold"]:
            x, inv_log_det_outer = self.outer_transform.inverse(h, full_jacobian=False, context=context if self.apply_context_to_outer else None)
            inv_jacobian_outer = None
        else:
            x, inv_jacobian_outer = self.outer_transform.inverse(h, full_jacobian=True, context=context if self.apply_context_to_outer else None)
            inv_log_det_outer = None

        return x, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer, h

    def _log_prob(self, mode, u, log_det_inner, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer):
        if mode == "mf":
            # inv_jacobian_outer is dx / du, but still need to restrict this to the manifold latents
            inv_jacobian_outer = inv_jacobian_outer[:, :, : self.latent_dim]
            # And finally calculate log det (J^T J)
            jtj = torch.bmm(torch.transpose(inv_jacobian_outer, -2, -1), inv_jacobian_outer)

            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob - 0.5 * torch.slogdet(jtj)[1] - inv_log_det_inner

        elif mode == "mf-fixed-manifold":
            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob - inv_log_det_inner

        else:
            log_prob = None

        return log_prob

    def _report_model_parameters(self):
        """ Reports the model size """
        super()._report_model_parameters()
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        inner_params = sum(p.numel() for p in self.inner_transform.parameters())
        outer_params = sum(p.numel() for p in self.outer_transform.parameters())
        logger.info("  Encoder:         %.1f M parameters", encoder_params / 1.0e06)
        logger.info("  Outer transform: %.1f M parameters", outer_params / 1.0e06)
        logger.info("  Inner transform: %.1f M parameters", inner_params / 1.0e06)
