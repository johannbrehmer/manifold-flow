#! /usr/bin/env python

"""
Small working example for the calculation of the Jacobian of an affine coupling transform. TL;DR: It's slow.

Tested with Python 3.6, JAX 0.1.46, and Matplotlib 3.1.1.

Strongly based on https://github.com/bayesiains/nsf, bugs are all my own though.
"""

import numpy as onp
import jax
import jax.numpy as np
from jax.experimental import stax
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax
import time
from matplotlib import pyplot as plt
from experiments.mwe import timer


def sum_except_batch(x):
    """Sums all elements of `x` except for the first dimension."""
    x = x.reshape((x.shape[0], -1))
    return np.sum(x, axis=1)


def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)


class AffineCouplingTransform:
    """An affine coupling layer that scales and shifts part of the variables.

    Supports 2D inputs (NxD), as well as 4D inputs for images (NxCxHxW). For images the splitting is done on the
    channel dimension, using the provided 1D mask.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(self, mask, hidden_features=100, hidden_layers=3, full_jacobian=False):
        timer.timer(start="__init__()")

        self.features = len(mask)
        self.identity_features = onp.argwhere(mask <= 0).flatten()
        self.transform_features = onp.argwhere(mask > 0).flatten()
        self.full_jacobian = full_jacobian

        assert len(self.identity_features) + len(self.transform_features) == self.features

        # Build transform net
        layers = [Dense(hidden_features), Relu]
        for _ in range(hidden_layers - 1):
            layers += [Dense(hidden_features), Relu]
        layers += [Dense(len(self.transform_features) * 2)]
        transform_net_init, self.transform_net_apply = stax.serial(*layers)

        # Initialize parameters, not committing to a batch shape
        rng = jax.random.PRNGKey(0)
        in_shape = (-1, len(self.identity_features))
        transform_net_out_shape, self.transform_net_params = transform_net_init(rng, in_shape)

        # Jacobian

        if full_jacobian == "forward":
            self._transform_jacobian = jax.jacfwd(self._transform)
        elif full_jacobian == "reverse":
            self._transform_jacobian = jax.jacrev(self._transform)
        else:
            self._transform_jacobian = None
        timer.timer(stop="__init__()")

    def _transform(self, inputs):
        """ Transformation in a separate function, making it easy to calculate the Jacobian """

        identity_split = inputs[:, self.identity_features]
        transform_split = inputs[:, self.transform_features]

        # Calculate transformation parameters based on first half of input
        transform_params = self.transform_net_apply(self.transform_net_params, identity_split)

        # Transform second half of input
        unconstrained_scale = transform_params[:, len(self.transform_features):, ...]
        shift = transform_params[:, :len(self.transform_features), ...]
        scale = sigmoid(unconstrained_scale + 2) + 1e-3
        transform_split = transform_split * scale + shift

        return transform_split

    def forward(self, inputs):
        # timer.timer(start="forward()")

        # Split input
        # timer.timer(start="  forward: split input")
        identity_split = inputs[:, self.identity_features]
        transform_split = inputs[:, self.transform_features]
        # timer.timer(stop="  forward: split input")

        # Calculate transformation parameters based on first half of input
        # timer.timer(start="  forward: transform params")
        transform_params = self.transform_net_apply(self.transform_net_params, identity_split)
        # timer.timer(stop="  forward: transform params")

        # Transform second half of input
        # timer.timer(start="  forward: transform input")
        unconstrained_scale = transform_params[:, len(self.transform_features):, ...]
        shift = transform_params[:, :len(self.transform_features), ...]
        scale = sigmoid(unconstrained_scale + 2) + 1e-3
        log_scale = np.log(scale)
        transform_split = transform_split * scale + shift
        # timer.timer(stop="  forward: transform input")

        # Merge outputs together
        # timer.timer(start="  forward: merge outputs")
        outputs = onp.empty_like(inputs)
        outputs[:, self.identity_features, ...] = identity_split
        outputs[:, self.transform_features, ...] = transform_split
        # jax.ops.index_update(outputs, jax.ops.index[:, self.identity_features], identity_split)
        # jax.ops.index_update(outputs, jax.ops.index[:, self.transform_features], transform_split)
        # timer.timer(stop="  forward: merge outputs")

        # Calculate full Jacobian matrix, or just log abs det Jacobian
        jacobian = None
        logabsdet = None
        if self.full_jacobian in ["forward", "reverse"]:
            # timer.timer(start="  forward: calculate jacobian")
            transform_jacobian = [
                self._transform_jacobian(input.reshape((1, input.shape[0])))
                for input in inputs
            ]
            transform_jacobian = onp.asarray(transform_jacobian)
            transform_jacobian = transform_jacobian.reshape(
                (transform_jacobian.shape[0], transform_jacobian.shape[2], transform_jacobian.shape[4])
            )
            # timer.timer(stop="  forward: calculate jacobian")
            # timer.timer(start="  forward: sum jacobian")
            # transform_jacobian = np.sum(transform_jacobian, axis=2)
            # timer.timer(stop="  forward: sum jacobian")

            # timer.timer(start="  forward: concatenate jacobian")
            jacobian = onp.zeros(inputs.shape + inputs.shape[1:])
            jacobian[:, self.identity_features, self.identity_features] = 1.
            jacobian[:, self.transform_features, :] = transform_jacobian
            # timer.timer(stop="  forward: concatenate jacobian")

            # # For debugging, check that the Jacobian determinant agrees
            # print("Determinant cross-check: {} vs {}".format(
            #     onp.linalg.slogdet(jacobian[0])[1],
            #     onp.sum(log_scale[0])
            # ))
        else:
            # timer.timer(start="  forward: sum logabsdet")
            logabsdet = sum_except_batch(log_scale)
            # timer.timer(stop="  forward: sum logabsdet")

        # timer.timer(stop="forward()")

        return outputs, jacobian if self.full_jacobian else logabsdet


def time_transform(features=100, batchsize=100, hidden_features=100, hidden_layers=5, calculate_full_jacobian=False):
    # TODO: CUDA

    # timer.reset()

    data = onp.random.randn(batchsize, features)
    mask = onp.zeros(features, dtype=np.bool_)
    mask[0::2] = 1
    transform = AffineCouplingTransform(mask, hidden_features=hidden_features, hidden_layers=hidden_layers, full_jacobian=calculate_full_jacobian)

    time_before = time.time()
    _ = transform.forward(data)
    time_taken = time.time() - time_before

    # timer.report_timer()

    return time_taken


def time_as_function_of_features(features=[2] + list(np.arange(50,251,25)), **kwargs):
    results_fwd = []
    results_rev = []
    results_det = []
    for feature in features:
        results_fwd.append(time_transform(features=feature, calculate_full_jacobian="forward", **kwargs))
        results_rev.append(time_transform(features=feature, calculate_full_jacobian="reverse", **kwargs))
        results_det.append(time_transform(features=feature, calculate_full_jacobian=False, **kwargs))
    return np.array(features), np.array(results_fwd), np.array(results_rev), np.array(results_det)


def time_as_function_of_batchsize(batchsizes=[1] + list(np.arange(50,251,25)), **kwargs):
    results_fwd = []
    results_rev = []
    results_det = []
    for batchsize in batchsizes:
        if batchsize < 1:
            batchsize = 1
        results_fwd.append(time_transform(batchsize=batchsize, calculate_full_jacobian="forward", **kwargs))
        results_rev.append(time_transform(batchsize=batchsize, calculate_full_jacobian="reverse", **kwargs))
        results_det.append(time_transform(batchsize=batchsize, calculate_full_jacobian=False, **kwargs))
    return np.array(batchsizes), np.array(results_fwd), np.array(results_rev), np.array(results_det)


def time_as_function_of_layers(layers=np.arange(1,21,1), **kwargs):
    results_fwd = []
    results_rev = []
    results_det = []
    for layer in layers:
        results_fwd.append(time_transform(hidden_layers=layer, calculate_full_jacobian="forward", **kwargs))
        results_rev.append(time_transform(hidden_layers=layer, calculate_full_jacobian="reverse", **kwargs))
        results_det.append(time_transform(hidden_layers=layer, calculate_full_jacobian=False, **kwargs))
    return np.array(layers), np.array(results_fwd), np.array(results_rev), np.array(results_det)


def plot_results(xs, ys_fwd, ys_rev, ys_det, labels, det_factor=1, filename="manifold_flow_timing_jax.pdf"):
    n_panels = len(xs)
    ymax = max([np.max(y) for y in ys_fwd] + [np.max(y) for y in ys_rev] + [det_factor * np.max(y) for y in ys_det]) * 1.05

    plt.figure(figsize=(4 * n_panels + 0.5, 4.5))

    for i, (x, y_fwd, y_rev, y_det, label) in enumerate(zip(xs, ys_fwd, ys_rev, ys_det, labels)):
        ax = plt.subplot(1, n_panels, i+1)

        plt.plot(x, y_fwd, c="C0", ls="-", label="Full Jacobian calculation, forward mode")
        plt.plot(x, y_rev, c="C1", ls="--", label="Full Jacobian calculation, reverse mode")
        plt.plot(x, det_factor * y_det, c="C2", ls=":", label=r"Determinant calculation")

        plt.xlabel(label)
        plt.ylim(0., ymax)
        if i == 0:
            plt.legend(loc="upper left")
            plt.ylabel("Time for forward pass [s]")
        else:
            plt.ylabel(None)
            ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(filename)


if __name__ == "__main__":
    print("Hi!")

    print("Simple timing test:")
    time_det = time_transform(calculate_full_jacobian=False)
    print("  Forward pass, calculating the Jacobian determinant:          {:.3f}s".format(time_det))
    time_fwd = time_transform(calculate_full_jacobian="forward")
    print("  Forward pass, calculating the full Jacobian in forward mode: {:.3f}s".format(time_fwd))
    time_rev = time_transform(calculate_full_jacobian="reverse")
    print("  Forward pass, calculating the full Jacobian in reverse mode: {:.3f}s".format(time_rev))

    print("Measuring time as function of features")
    x_features, y_fwd_features, y_rev_features, y_det_features = time_as_function_of_features()
    print("Measuring time as function of batch size")
    x_batchsize, y_fwd_batchsize, y_rev_batchsize, y_det_batchsize = time_as_function_of_batchsize()
    print("Measuring time as function of hidden layers")
    x_hidden, y_fwd_hidden, y_rev_hidden, y_det_hidden = time_as_function_of_layers()
    print("Saving results to manifold_flow_timing.pdf")
    plot_results(
        xs = [x_features, x_batchsize, x_hidden],
        ys_fwd = [y_fwd_features, y_fwd_batchsize, y_fwd_hidden],
        ys_rev = [y_rev_features, y_rev_batchsize, y_rev_hidden],
        ys_det = [y_det_features, y_det_batchsize, y_det_hidden],
        labels = ["Data features", "Batch size", "Hidden layers"]
    )

    print("That's it, have a nice day!")
