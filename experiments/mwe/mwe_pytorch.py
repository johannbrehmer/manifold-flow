"""
Small working example for the calculation of the Jacobian of an affine coupling transform. TL;DR: It's slow.

Requires utils_pytorch.py and pyTorch >= 1.2.

Strongly based on https://github.com/bayesiains/nsf, bugs are all my own though.
"""

import numpy as np
import torch
from torch import nn
import time


def calculate_jacobian(outputs, inputs, create_graph=True):
    """Computes the jacobian of outputs with respect to inputs.

    Based on gelijergensen's code at https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa.

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """
    jac = outputs.new_zeros(outputs.size() + inputs.size()).view((-1,) + inputs.size())
    for i, out in enumerate(outputs.view(-1)):
        col_i = torch.autograd.grad(out, inputs, create_graph=create_graph, allow_unused=True, only_inputs=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()

    return jac.view(outputs.size() + inputs.size())


def batch_jacobian(outputs, inputs, create_graph=True):
    """ Batches calculate_jacobian."""

    # Strategy A: not very efficient, since we know there are no cross-batch effects.
    jacs = calculate_jacobian(outputs, inputs)
    jacs = jacs.view((outputs.size(0), np.prod(outputs.size()[1:]), inputs.size(0), np.prod(inputs.size()[1:]), ))
    jacs = torch.einsum("bibj->bij", jacs)

    # # Strategy B: equally inefficient...
    # jacs = []
    # for i in range(outputs.size(0)):
    #     jac = calculate_jacobian(outputs[i], inputs)
    #     jacs.append(jac.unsqueeze(0)[:,:,i,:])
    # jacs = torch.cat(jacs, 0)

    # # Strategy C: Faster, but doesn't work (d outputs / d inputs[i] isn't defined, only d outputs / d inputs)
    # jacs = []
    # for output, input in zip(outputs, inputs):
    #     jac = calculate_jacobian(output, input)  # This is 0 :(
    #     jacs.append(jac.unsqueeze(0))
    # jacs = torch.cat(jacs, 0)

    return jacs


def batch_diagonal(input):
    """ Batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N) """
    # make a zero matrix, which duplicates the last dim of input
    dims = [input.size(i) for i in torch.arange(input.dim())]
    dims.append(dims[-1])
    output = torch.zeros(dims)
    # stride across the first dimensions, add one to get the diagonal of the last dimension
    strides = [output.stride(i) for i in torch.arange(input.dim() - 1)]
    strides.append(output.size(-1) + 1)
    # stride and copy the imput to the diagonal
    output.as_strided(input.size(), strides).copy_(input)
    return output


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


class AffineCouplingTransform(nn.Module):
    """An affine coupling layer that scales and shifts part of the variables.

    Supports 2D inputs (NxD), as well as 4D inputs for images (NxCxHxW). For images the splitting is done on the
    channel dimension, using the provided 1D mask.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(self, mask, hidden_features=100, hidden_layers=3):
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

        layers = [nn.Linear(len(self.identity_features), hidden_features), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_features, hidden_features), nn.ReLU()]
        layers += [nn.Linear(hidden_features, len(self.transform_features) * 2)]
        self.transform_net = nn.Sequential(*layers)

    def forward(self, inputs, full_jacobian=False):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Inputs must be a 2D or a 4D tensor.')
        if inputs.shape[1] != self.features:
            raise ValueError('Expected features = {}, got {}.'.format(
                self.features, inputs.shape[1]))

        # Split input
        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        # Calculate transformation parameters based on first half of input
        transform_params = self.transform_net(identity_split)

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
            jacobian = torch.zeros(inputs.size() + inputs.size()[1:])
            jacobian[:, self.identity_features, self.identity_features] = 1.
            jacobian[:, self.transform_features, :] = batch_jacobian(transform_split, inputs)

            # # For debugging, check that the Jacobian determinant agrees
            # print("Determinant cross-check: {} vs {}".format(
            #     torch.slogdet(jacobian[0]),
            #     torch.sum(log_scale[0])
            # ))
        else:
            logabsdet = sum_except_batch(log_scale, num_batch_dims=1)

        return outputs, jacobian if full_jacobian else logabsdet


def time_transform(features=100, batchsize=100, hidden_features=100, hidden_layers=10, calculate_full_jacobian=True):
    data = torch.randn(batchsize, features)
    data.requires_grad = True

    mask = torch.zeros(features).byte()
    mask[0::2] += 1

    transform = AffineCouplingTransform(mask, hidden_features=hidden_features, hidden_layers=hidden_layers)

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
