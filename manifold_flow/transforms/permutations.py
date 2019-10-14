"""Implementations of permutation-like transforms."""

import torch

from manifold_flow import utils, transforms


class Permutation(transforms.Transform):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, permutation, dim=1):
        if permutation.ndimension() != 1:
            raise ValueError('Permutation must be a 1D tensor.')
        if not utils.is_positive_int(dim):
            raise ValueError('dim must be a positive integer.')

        super().__init__()
        self._dim = dim
        self.register_buffer('_permutation', permutation)

    @property
    def _inverse_permutation(self):
        return torch.argsort(self._permutation)

    @staticmethod
    def _permute(inputs, permutation, dim, full_jacobian=False):
        if dim >= inputs.ndimension():
            raise ValueError("No dimension {} in inputs.".format(dim))
        if inputs.shape[dim] != len(permutation):
            raise ValueError("Dimension {} in inputs must be of size {}."
                             .format(dim, len(permutation)))
        batch_size = inputs.shape[0]
        if full_jacobian:
            inputs.requires_grad = True
            outputs = torch.index_select(inputs, dim, permutation)
            jacobian = utils.batch_jacobian(outputs, inputs)
            return outputs, jacobian
        else:
            outputs = torch.index_select(inputs, dim, permutation)
            logabsdet = torch.zeros(batch_size)
            return outputs, logabsdet

    def forward(self, inputs, context=None, full_jacobian=False):
        return self._permute(inputs, self._permutation, self._dim, full_jacobian=full_jacobian)

    def inverse(self, inputs, context=None, full_jacobian=False):
        return self._permute(inputs, self._inverse_permutation, self._dim, full_jacobian=full_jacobian)


class RandomPermutation(Permutation):
    """Permutes using a random, but fixed, permutation. Only works with 1D inputs."""

    def __init__(self, features, dim=1):
        if not utils.is_positive_int(features):
            raise ValueError('Number of features must be a positive integer.')
        super().__init__(torch.randperm(features), dim)


class ReversePermutation(Permutation):
    """Reverses the elements of the input. Only works with 1D inputs."""

    def __init__(self, features, dim=1):
        if not utils.is_positive_int(features):
            raise ValueError('Number of features must be a positive integer.')
        super().__init__(torch.arange(features - 1, -1, -1), dim)
