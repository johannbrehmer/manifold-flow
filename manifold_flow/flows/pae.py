import logging

from manifold_flow.utils.various import product
from manifold_flow import distributions, transforms
from manifold_flow.flows import BaseFlow

logger = logging.getLogger(__name__)


class ProbabilisticAutoEncoder(BaseFlow):
    """ Probabilistic Auto-Encoder = encoder + decoder + flow in latent space. See 2006.05479 """

    def __init__(self, data_dim, latent_dim, encoder, decoder, inner_transform=None, apply_context_to_outer=True):
        super(ProbabilisticAutoEncoder, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.total_data_dim = product(data_dim)
        self.total_latent_dim = product(latent_dim)
        self.apply_context_to_outer = apply_context_to_outer

        assert self.total_latent_dim < self.total_data_dim

        self.manifold_latent_distribution = distributions.StandardNormal((self.total_latent_dim,))

        self.encoder = encoder
        self.decoder = decoder
        if inner_transform is None:
            self.inner_transform = transforms.IdentityTransform()
        else:
            self.inner_transform = inner_transform

        self._report_model_parameters()

    def forward(self, x, context=None, return_hidden=False):
        """
        Transforms data point to latent space, evaluates likelihood, and transforms it back to data space.
        """

        # Encode
        u, h_manifold, log_det_inner = self._encode(x, context=context)

        # Decode
        x_reco = self._decode(u, context=context)

        # Log prob
        log_prob = self._log_prob(u, log_det_inner)

        if return_hidden:
            return x_reco, log_prob, u, h_manifold
        return x_reco, log_prob, u

    def encode(self, x, context=None):
        """ Transforms data point to latent space. """
        return self._encode(x, context=context)[0]

    def decode(self, u, context=None):
        """ Decodes latent variable to data space."""
        return self._decode(u, context=context)

    def log_prob(self, x, context=None):
        """ Evaluates log likelihood for given data point."""
        return self.forward(x, context=context)[1]

    def sample(self, u=None, n=1, context=None):
        """
        Generates samples from model.

        Note: this is PIE / MF sampling! Cannot sample from slice of PIE efficiently.
        """

        if u is None:
            u = self.manifold_latent_distribution.sample(n, context=None)
        x = self.decode(u, context=context)
        return x

    def _encode(self, x, context=None):
        h_manifold = self.encoder(x, context=context if self.apply_context_to_outer else None)
        u, log_det_inner = self.inner_transform(h_manifold, full_jacobian=False, context=context)

        return u, h_manifold, log_det_inner

    def _decode(self, u, context=None):
        h, _ = self.inner_transform.inverse(u, full_jacobian=False, context=context)
        x = self.decoder(h, context=context if self.apply_context_to_outer else None)

        return x

    def _log_prob(self, mode, u, log_det_inner):
        return self.manifold_latent_distribution._log_prob(u, context=None) + log_det_inner

    def _report_model_parameters(self):
        """ Reports the model size """
        super()._report_model_parameters()

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        inner_params = sum(p.numel() for p in self.inner_transform.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        logger.info("  Encoder:         %.1f M parameters", encoder_params / 1.0e06)
        logger.info("  Decoder:         %.1f M parameters", decoder_params / 1.0e06)
        logger.info("  Inner transform: %.1f M parameters", inner_params / 1.0e06)
