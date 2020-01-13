"""Implementations of Normal distributions."""

import numpy as np
import torch

from manifold_flow import distributions
from manifold_flow.utils import various


class StandardNormal(distributions.Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super().__init__()
        self._shape = torch.Size(shape)
        self._log_z = 0.5 * np.prod(shape) * np.log(2 * np.pi)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError("Expected input of shape {}, got {}".format(self._shape, inputs.shape[1:]))
        neg_energy = -0.5 * various.sum_except_batch(inputs ** 2, num_batch_dims=1)
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            return torch.randn(num_samples, *self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            context_size = context.shape[0]
            samples = torch.randn(context_size * num_samples, *self._shape)
            return various.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return torch.zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return torch.zeros(context.shape[0], *self._shape)


class RescaledNormal(distributions.Distribution):
    """A multivariate Normal with zero mean and a diagonal covariance that is epsilon^2 along each diagonal entry of the matrix."""

    def __init__(self, shape, std=1.0, clip=10.0):
        super().__init__()
        self._shape = torch.Size(shape)
        self.std = std
        self._clip = clip
        self._log_z = 0.5 * np.prod(shape) * np.log(2 * np.pi) + np.prod(shape) * np.log(self.std)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError("Expected input of shape {}, got {}".format(self._shape, inputs.shape[1:]))
        inputs = torch.clamp(inputs, -self._clip, self._clip)
        neg_energy = -0.5 * various.sum_except_batch(inputs ** 2, num_batch_dims=1) / self.std ** 2
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            return self.std * torch.randn(num_samples, *self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            context_size = context.shape[0]
            samples = self.std * torch.randn(context_size * num_samples, *self._shape)
            return various.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return torch.zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return torch.zeros(context.shape[0], *self._shape)


class DiagonalNormal(distributions.Distribution):
    """A multivariate Normal with zero mean and a flexible diagonal covariance matrix."""

    def __init__(self, shape, initial_std=1.0, clip=10.0):
        super().__init__()
        self._shape = torch.Size(shape)
        self._clip = clip
        self._log_z_constant = 0.5 * np.prod(shape) * np.log(2 * np.pi)

        self.log_stds = torch.nn.Parameter(np.log(initial_std) * torch.ones(shape=self._shape))

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError("Expected input of shape {}, got {}".format(self._shape, inputs.shape[1:]))
        inputs = torch.clamp(inputs, -self._clip, self._clip)

        stds = torch.exp(self.log_stds).unsqueeze(0)
        neg_energy = -0.5 * various.sum_except_batch(inputs ** 2 / stds ** 2, num_batch_dims=1)
        log_z = self._log_z_constant + torch.sum(self.log_stds)
        return neg_energy - log_z

    def _sample(self, num_samples, context):
        if context is None:
            return self.std * torch.randn(num_samples, *self._shape)
        else:
            stds = torch.exp(self.log_stds).unsqueeze(0)
            # The value of the context is ignored, only its size is taken into account.
            context_size = context.shape[0]

            samples = torch.randn(context_size * num_samples, *self._shape)
            samples = various.split_leading_dim(samples, [context_size, num_samples])
            samples = stds * samples

            return samples

    def _mean(self, context):
        if context is None:
            return torch.zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return torch.zeros(context.shape[0], *self._shape)


class ConditionalDiagonalNormal(distributions.Distribution):
    """A diagonal multivariate Normal whose parameters are functions of a context."""

    def __init__(self, shape, context_encoder=None):
        """Constructor.

        Args:
            shape: list, tuple or torch.Size, the shape of the input variables.
            context_encoder: callable or None, encodes the context to the distribution parameters.
                If None, defaults to the identity function.
        """
        super().__init__()
        self._shape = torch.Size(shape)
        if context_encoder is None:
            self._context_encoder = lambda x: x
        else:
            self._context_encoder = context_encoder
        self._log_z = 0.5 * np.prod(shape) * np.log(2 * np.pi)

    def _compute_params(self, context):
        """Compute the means and log stds form the context."""
        if context is None:
            raise ValueError("Context can't be None.")

        params = self._context_encoder(context)
        if params.shape[-1] % 2 != 0:
            raise RuntimeError("The context encoder must return a tensor whose last dimension is even.")
        if params.shape[0] != context.shape[0]:
            raise RuntimeError("The batch dimension of the parameters is inconsistent with the input.")

        split = params.shape[-1] // 2
        means = params[..., :split].reshape(params.shape[0], *self._shape)
        log_stds = params[..., split:].reshape(params.shape[0], *self._shape)
        return means, log_stds

    def _log_prob(self, inputs, context):
        if inputs.shape[1:] != self._shape:
            raise ValueError("Expected input of shape {}, got {}".format(self._shape, inputs.shape[1:]))

        # Compute parameters.
        means, log_stds = self._compute_params(context)
        assert means.shape == inputs.shape and log_stds.shape == inputs.shape

        # Compute log prob.
        norm_inputs = (inputs - means) * torch.exp(-log_stds)
        log_prob = -0.5 * various.sum_except_batch(norm_inputs ** 2, num_batch_dims=1)
        log_prob -= various.sum_except_batch(log_stds, num_batch_dims=1)
        log_prob -= self._log_z
        return log_prob

    def _sample(self, num_samples, context):
        # Compute parameters.
        means, log_stds = self._compute_params(context)
        stds = torch.exp(log_stds)
        means = various.repeat_rows(means, num_samples)
        stds = various.repeat_rows(stds, num_samples)

        # Generate samples.
        context_size = context.shape[0]
        noise = torch.randn(context_size * num_samples, *self._shape)
        samples = means + stds * noise
        return various.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        means, _ = self._compute_params(context)
        return means
