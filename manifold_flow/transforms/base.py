"""Basic definitions for the transforms module."""

import numpy as np
import torch
from torch import nn
import logging

from manifold_flow.utils import various

logger = logging.getLogger(__name__)


class InverseNotAvailable(Exception):
    """Exception to be thrown when a transform does not have an inverse."""

    pass


class InputOutsideDomain(Exception):
    """Exception to be thrown when the input to a transform is not within its domain."""

    pass


class Transform(nn.Module):
    """Base class for all transform objects."""

    def forward(self, inputs, context=None, full_jacobian=False):
        raise NotImplementedError()

    def inverse(self, inputs, context=None, full_jacobian=False):
        raise InverseNotAvailable()


class CompositeTransform(Transform):
    """Composes several transforms into one, in the order they are given."""

    def __init__(self, transforms):
        """Constructor.

        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__()
        self._transforms = nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, funcs, context, full_jacobian=False):
        batch_size = inputs.shape[0]
        outputs = inputs

        if full_jacobian:
            total_jacobian = None
            for func in funcs:
                inputs = outputs
                outputs, jacobian = func(inputs, context, full_jacobian=True)

                # # Cross-check for debugging
                # _, logabsdet = func(inputs, context, full_jacobian=False)
                # _, logabsdet_from_jacobian = torch.slogdet(jacobian)
                # logger.debug("Transformation %s has Jacobian\n%s\nwith log abs det %s (ground truth %s)", type(func).__name__, jacobian.detach().numpy()[0], logabsdet_from_jacobian[0].item(), logabsdet[0].item())

                # timer.timer(start="Jacobian multiplication")
                total_jacobian = jacobian if total_jacobian is None else torch.bmm(jacobian, total_jacobian)
                # timer.timer(stop="Jacobian multiplication")

            # logger.debug("Composite Jacobians \n %s", total_jacobian[0])

            return outputs, total_jacobian

        else:
            total_logabsdet = torch.zeros(batch_size)
            for func in funcs:
                outputs, logabsdet = func(outputs, context)
                total_logabsdet += logabsdet
            return outputs, total_logabsdet

    def forward(self, inputs, context=None, full_jacobian=False):
        funcs = self._transforms
        return self._cascade(inputs, funcs, context, full_jacobian)

    def inverse(self, inputs, context=None, full_jacobian=False):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context, full_jacobian)


class MultiscaleCompositeTransform(Transform):
    """A multiscale composite transform as described in the RealNVP paper.

    Splits the outputs along the given dimension after every transform, outputs one half, and
    passes the other half to further transforms. No splitting is done before the last transform.

    Note: Inputs could be of arbitrary shape, but outputs will always be flattened.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(self, num_transforms, split_dim=1):
        """Constructor.

        Args:
            num_transforms: int, total number of transforms to be added.
            split_dim: dimension along which to split.
        """
        if not various.is_positive_int(split_dim):
            raise TypeError("Split dimension must be a positive integer.")

        super().__init__()
        self._transforms = nn.ModuleList()
        self._output_shapes = []
        self._num_transforms = num_transforms
        self._split_dim = split_dim

    def add_transform(self, transform, transform_output_shape):
        """Add a transform. Must be called exactly `num_transforms` times.

        Parameters:
            transform: the `Transform` object to be added.
            transform_output_shape: tuple, shape of transform's outputs, excl. the first batch
                dimension.

        Returns:
            Input shape for the next transform, or None if adding the last transform.
        """
        assert len(self._transforms) <= self._num_transforms

        if len(self._transforms) == self._num_transforms:
            raise RuntimeError("Adding more than {} transforms is not allowed.".format(self._num_transforms))

        if (self._split_dim - 1) >= len(transform_output_shape):
            raise ValueError("No split_dim in output shape")

        if transform_output_shape[self._split_dim - 1] < 2:
            raise ValueError("Size of dimension {} must be at least 2.".format(self._split_dim))

        self._transforms.append(transform)

        if len(self._transforms) != self._num_transforms:  # Unless last transform.
            output_shape = list(transform_output_shape)
            output_shape[self._split_dim - 1] = (output_shape[self._split_dim - 1] + 1) // 2
            output_shape = tuple(output_shape)

            hidden_shape = list(transform_output_shape)
            hidden_shape[self._split_dim - 1] = hidden_shape[self._split_dim - 1] // 2
            hidden_shape = tuple(hidden_shape)
        else:
            # No splitting for last transform.
            output_shape = transform_output_shape
            hidden_shape = None

        self._output_shapes.append(output_shape)
        return hidden_shape

    def forward(self, inputs, context=None, full_jacobian=False):
        if self._split_dim >= inputs.dim():
            raise ValueError("No split_dim in inputs.")
        if self._num_transforms != len(self._transforms):
            raise RuntimeError("Expecting exactly {} transform(s) " "to be added.".format(self._num_transforms))

        batch_size = inputs.shape[0]

        def cascade():
            hiddens = inputs

            for i, transform in enumerate(self._transforms[:-1]):
                transform_outputs, logabsdet = transform(hiddens, context, full_jacobian)
                outputs, hiddens = torch.chunk(transform_outputs, chunks=2, dim=self._split_dim)
                assert outputs.shape[1:] == self._output_shapes[i]
                yield outputs, logabsdet

            # Don't do the splitting for the last transform.
            outputs, logabsdet = self._transforms[-1](hiddens, context, full_jacobian)
            yield outputs, logabsdet

        if full_jacobian:
            all_outputs = []
            total_jacobian = None

            for outputs, jacobian in cascade():
                all_outputs.append(outputs.reshape(batch_size, -1))
                total_jacobian = jacobian if total_jacobian is None else torch.mm(jacobian, total_jacobian)

            all_outputs = torch.cat(all_outputs, dim=-1)
            return all_outputs, total_jacobian

        else:
            all_outputs = []
            total_logabsdet = torch.zeros(batch_size)

            for outputs, logabsdet in cascade():
                all_outputs.append(outputs.reshape(batch_size, -1))
                total_logabsdet += logabsdet

            all_outputs = torch.cat(all_outputs, dim=-1)
            return all_outputs, total_logabsdet

    def inverse(self, inputs, context=None, full_jacobian=False):
        if inputs.dim() != 2:
            raise ValueError("Expecting NxD inputs")
        if self._num_transforms != len(self._transforms):
            raise RuntimeError("Expecting exactly {} transform(s) " "to be added.".format(self._num_transforms))

        batch_size = inputs.shape[0]

        rev_inv_transforms = [transform.inverse for transform in self._transforms[::-1]]

        split_indices = np.cumsum([np.prod(shape) for shape in self._output_shapes])
        split_indices = np.insert(split_indices, 0, 0)

        split_inputs = []
        for i in range(len(self._output_shapes)):
            flat_input = inputs[:, split_indices[i] : split_indices[i + 1]]
            split_inputs.append(flat_input.view(-1, *self._output_shapes[i]))
        rev_split_inputs = split_inputs[::-1]

        if full_jacobian:

            # We don't do the splitting for the last (here first) transform.
            hiddens, total_jacobian = rev_inv_transforms[0](rev_split_inputs[0], context, full_jacobian=True)

            for inv_transform, input_chunk in zip(rev_inv_transforms[1:], rev_split_inputs[1:]):
                tmp_concat_inputs = torch.cat([input_chunk, hiddens], dim=self._split_dim)
                hiddens, jacobian = inv_transform(tmp_concat_inputs, context, full_jacobian=True)
                total_jacobian = torch.mm(jacobian, total_jacobian)

            outputs = hiddens

            return outputs, total_jacobian

        else:
            total_logabsdet = torch.zeros(batch_size)

            # We don't do the splitting for the last (here first) transform.
            hiddens, logabsdet = rev_inv_transforms[0](rev_split_inputs[0], context)
            total_logabsdet += logabsdet

            for inv_transform, input_chunk in zip(rev_inv_transforms[1:], rev_split_inputs[1:]):
                tmp_concat_inputs = torch.cat([input_chunk, hiddens], dim=self._split_dim)
                hiddens, logabsdet = inv_transform(tmp_concat_inputs, context)
                total_logabsdet += logabsdet

            outputs = hiddens

            return outputs, total_logabsdet


class InverseTransform(Transform):
    """Creates a transform that is the inverse of a given transform."""

    def __init__(self, transform):
        """Constructor.

        Args:
            transform: An object of type `Transform`.
        """
        super().__init__()
        self._transform = transform

    def forward(self, inputs, context=None, full_jacobian=False):
        return self._transform.inverse(inputs, context, full_jacobian)

    def inverse(self, inputs, context=None, full_jacobian=False):
        return self._transform(inputs, context, full_jacobian)
