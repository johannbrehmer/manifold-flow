from torch import nn
import logging

logger = logging.getLogger(__name__)


class BaseFlow(nn.Module):
    """ Abstract base flow class """

    def forward(self, x, context=None):
        raise NotImplementedError

    def encode(self, x, context=None):
        raise NotImplementedError

    def decode(self, u, context=None):
        raise NotImplementedError

    def project(self, x, context=None):
        return self.decode(self.encode(x, context), context)

    def log_prob(self, x, context=None):
        raise NotImplementedError

    def sample(self, u=None, n=1, context=None):
        raise NotImplementedError

    def _report_model_parameters(self):
        """ Reports the model size """

        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size = all_params * (32 / 8)  # Bytes
        logger.info(
            "Model has %.1f M parameters (%.1f M trainable) with an estimated size of %.1f MB", all_params / 1e6, trainable_params / 1.0e6, size / 1.0e6
        )
