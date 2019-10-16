"""
Small working example for the calculation of the Jacobian of an affine coupling transform. TL;DR: It's slow.

Requires utils_pytorch.py and pyTorch >= 1.2.

Strongly based on https://github.com/bayesiains/nsf, bugs are all my own though.
"""

import torch
import sys
from torch import nn
import time

sys.path.append(".")
from utils_pytorch import batch_jacobian, sum_except_batch, ResidualNet


class AffineCouplingTransform(nn.Module):
    """An affine coupling layer that scales and shifts part of the variables.

    Supports 2D inputs (NxD), as well as 4D inputs for images (NxCxHxW). For images the splitting is done on the
    channel dimension, using the provided 1D mask.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(self,
                 mask,
                 hidden_features=20,
                 num_blocks=5,
                 activation=nn.functional.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        """
        Constructor.

        Args:
            mask: a 1-dim tensor, tuple or list. It indexes inputs as follows:
                * If `mask[i] > 0`, `input[i]` will be transformed.
                * If `mask[i] <= 0`, `input[i]` will be passed unchanged.
        """

        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError('Mask must be a 1-dim tensor.')
        if mask.numel() <= 0:
            raise ValueError('Mask can\'t be empty.')

        super().__init__()
        self.features = len(mask)
        features_vector = torch.arange(self.features)

        self.register_buffer('identity_features', features_vector.masked_select(mask <= 0))
        self.register_buffer('transform_features', features_vector.masked_select(mask > 0))

        assert len(self.identity_features) + len(self.transform_features) == self.features

        self.transform_net = ResidualNet(
            in_features=len(self.identity_features),
            out_features=len(self.transform_features) * 2,
            hidden_features=hidden_features,
            context_features=None,
            num_blocks=num_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, inputs, context=None, full_jacobian=False):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Inputs must be a 2D or a 4D tensor.')
        if inputs.shape[1] != self.features:
            raise ValueError('Expected features = {}, got {}.'.format(
                self.features, inputs.shape[1]))

        # Split input
        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        # Calculate transformation parameters based on first half of input
        transform_params = self.transform_net(identity_split, context)

        # Transform second half of input
        unconstrained_scale = transform_params[:, len(self.transform_features):, ...]
        shift = transform_params[:, :len(self.transform_features), ...]
        scale = torch.sigmoid(unconstrained_scale + 2) + 1e-3
        log_scale = torch.log(scale)
        transform_split = transform_split * scale + shift

        # Merge outputs together
        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features, ...] = identity_split
        outputs[:, self.transform_features, ...] = transform_split

        # Calculate full Jacobian matrix, or just log abs det Jacobian
        jacobian = None
        logabsdet = None
        if full_jacobian:
            jacobian_transform = batch_jacobian(transform_split, inputs)
            jacobian_identity = torch.eye(len(self.identity_features)).unsqueeze(0)  # (1, n, n)
            jacobian = torch.zeros(inputs.size() + inputs.size()[1:])
            (jacobian[:, self.identity_features, :])[:, :, self.identity_features] = jacobian_identity
            jacobian[:, self.transform_features, :] = jacobian_transform
        else:
            logabsdet = sum_except_batch(log_scale, num_batch_dims=1)

        return outputs, jacobian if full_jacobian else logabsdet


def time_transform(features=20, batchsize=100, hidden_features=100, num_blocks=10, calculate_full_jacobian=True):
    data = torch.randn(batchsize, features)
    data.requires_grad = True

    mask = torch.zeros(features).byte()
    mask[0::2] += 1

    transform = AffineCouplingTransform(mask, hidden_features=hidden_features, num_blocks=num_blocks)

    time_before = time.time()
    _ = transform(data, full_jacobian=calculate_full_jacobian)
    time_taken = time.time() - time_before

    return time_taken


if __name__ == "__main__":
    print("Hi!")
    time_det = time_transform(calculate_full_jacobian=False)
    print("Forward pass, calculating the Jacobian determinant: {:.3f}s".format(time_det))
    time_full = time_transform(calculate_full_jacobian=True)
    print("Forward pass, calculating the full Jacobian:        {:.3f}s".format(time_full))
    print("That's it, have a nice day!")
