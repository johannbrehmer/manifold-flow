from __future__ import absolute_import, division, print_function, unicode_literals

import six
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def shuffle(*arrays):
    """ Shuffles multiple arrays simultaneously """

    permutation = None
    n_samples = None
    shuffled_arrays = []

    for i, a in enumerate(arrays):
        if a is None:
            shuffled_arrays.append(a)
            continue

        if permutation is None:
            n_samples = a.shape[0]
            permutation = np.random.permutation(n_samples)

        assert a.shape[0] == n_samples
        shuffled_a = a[permutation]
        shuffled_arrays.append(shuffled_a)

    return shuffled_arrays


def sanitize_array(
    array,
    replace_nan=0.0,
    replace_inf=0.0,
    replace_neg_inf=0.0,
    min_value=None,
    max_value=None,
):
    array[np.isneginf(array)] = replace_neg_inf
    array[np.isinf(array)] = replace_inf
    array[np.isnan(array)] = replace_nan

    if min_value is not None or max_value is not None:
        array = np.clip(array, min_value, max_value)

    return array


def weighted_quantile(
    values, quantiles, sample_weight=None, values_sorted=False, old_style=False
):
    """
    Calculates quantiles (similar to np.percentile), but supports weights.

    Parameters
    ----------
    values : ndarray
        Data
    quantiles : ndarray
        Which quantiles to calculate
    sample_weight : ndarray or None
        Weights
    values_sorted : bool
        If True, will avoid sorting the initial array
    old_style : bool
        If True, will correct output to be consistent with np.percentile

    Returns
    -------
    quantiles : ndarray
        Quantiles

    """

    # Input
    values = np.array(values, dtype=np.float64)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight, dtype=np.float64)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    # Sort
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    # Quantiles
    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight

    # Postprocessing
    if old_style:
        # To be consistent with np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)

    return np.interp(quantiles, weighted_quantiles, values)


def approx_equal(a, b, epsilon=1.0e-6):
    return abs(a - b) < epsilon


def create_missing_folders(folders):
    if folders is None:
        return

    for folder in folders:
        if folder is None or folder == "":
            continue

        if not os.path.exists(folder):
            os.makedirs(folder)

        elif not os.path.isdir(folder):
            raise OSError("Path {} exists, but is no directory!".format(folder))


def product(x):
    try:
        prod = 1
        for factor in x:
            prod *= factor
        return prod
    except:
        return x
