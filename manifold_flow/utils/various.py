import math
import os
import numpy as np
import logging
import torch

logger = logging.getLogger(__name__)


def check_dependence(outputs, inputs):
    try:
        for i, out in enumerate(outputs.view(-1)):
            col_i = torch.autograd.grad(out, inputs, retain_graph=True, create_graph=False, allow_unused=True)[0]
            if col_i is not None and np.max(np.abs(col_i.detach().numpy())) > 1.0e-9:
                return True
        return False
    except RuntimeError as e:
        logger.debug("%s", e)
        return False


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
        col_i = torch.autograd.grad(out, inputs, retain_graph=True, create_graph=create_graph, allow_unused=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()

    return jac.view(outputs.size() + inputs.size())


#
# def batch_jacobian(outputs, inputs, create_graph=True):
#     """Computes the jacobian of outputs with respect to inputs, assuming the first dimension of both are the minibatch.
#
#     Based on gelijergensen's code at https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa.
#
#     :param outputs: tensor for the output of some function
#     :param inputs: tensor for the input of some function (probably a vector)
#     :param create_graph: set True for the resulting jacobian to be differentible
#     :returns: a tensor of size (outputs.size() + inputs.size()) containing the
#         jacobian of outputs with respect to inputs
#     """
#
#     jacs = []
#     for input, output in zip(inputs, outputs):
#         jacs.append(calculate_jacobian(output, input, create_graph).unsqueeze(0))  # DOESN'T WORK
#     jacs = torch.cat(jacs, 0)
#     return jacs


def batch_jacobian(outputs, inputs, create_graph=True):
    """Computes the jacobian of outputs with respect to inputs, assuming the first dimension of both are the minibatch.

    Based on gelijergensen's code at https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa.

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """

    jac = calculate_jacobian(outputs, inputs)
    jac = jac.view((outputs.size(0), np.prod(outputs.size()[1:]), inputs.size(0), np.prod(inputs.size()[1:])))
    jac = torch.einsum("bibj->bij", jac)

    if create_graph:
        jac.requires_grad_()

    return jac


def batch_diagonal(input):
    # idea from here: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
    # batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N)
    # works in  2D -> 3D, should also work in higher dimensions
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


def sanitize_array(array, replace_nan=0.0, replace_inf=0.0, replace_neg_inf=0.0, min_value=None, max_value=None):
    array[np.isneginf(array)] = replace_neg_inf
    array[np.isinf(array)] = replace_inf
    array[np.isnan(array)] = replace_nan

    if min_value is not None or max_value is not None:
        array = np.clip(array, min_value, max_value)

    return array


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
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
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), "quantiles should be in [0, 1]"

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


def make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError("tensor or list of tensors expected, got {}".format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding).narrow(2, x * width + padding, width - padding).copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image

    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def tile(x, n):
    if not is_positive_int(n):
        raise TypeError("Argument 'n' must be a positive integer.")
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if not is_nonnegative_int(num_batch_dims):
        raise TypeError("Number of batch dimensions must be a non-negative integer.")
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


def merge_leading_dims(x, num_dims):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""
    if not is_positive_int(num_dims):
        raise TypeError("Number of leading dims must be a positive integer.")
    if num_dims > x.dim():
        raise ValueError("Number of leading dims can't be greater than total number of dims.")
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    if not is_positive_int(num_reps):
        raise TypeError("Number of repetitions must be a positive integer.")
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)


def tensor2numpy(x):
    return x.detach().cpu().numpy()


def logabsdet(x):
    """Returns the log absolute determinant of square matrix x."""
    # Note: torch.logdet() only works for positive determinant.
    _, res = torch.slogdet(x)
    return res


def random_orthogonal(size):
    """
    Returns a random orthogonal matrix as a 2-dim tensor of shape [size, size].
    """

    # Use the QR decomposition of a random Gaussian matrix.
    x = torch.randn(size, size)
    q, _ = torch.qr(x)
    return q


def get_num_parameters(model):
    """
    Returns the number of trainable parameters in a model of type nn.Module
    :param model: nn.Module containing trainable parameters
    :return: number of trainable parameters in model
    """
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters


def create_split_binary_mask(features, n_active):
    """
    Creates a binary mask of a given dimension in which the first n_active features are set to 1 and the others to 0.

    :param features: Dimension of mask.
    :param even: If True, even values are assigned 1s, odd 0s. If False, vice versa.
    :return: Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    mask[:n_active] += 1
    return mask


def create_alternating_binary_mask(features, even=True):
    """
    Creates a binary mask of a given dimension which alternates its masking.

    :param features: Dimension of mask.
    :param even: If True, even values are assigned 1s, odd 0s. If False, vice versa.
    :return: Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


def create_mid_split_binary_mask(features):
    """
    Creates a binary mask of a given dimension which splits its masking at the midpoint.

    :param features: Dimension of mask.
    :return: Binary mask split at midpoint of type torch.Tensor
    """
    mask = torch.zeros(features).byte()
    midpoint = features // 2 if features % 2 == 0 else features // 2 + 1
    mask[:midpoint] += 1
    return mask


def create_random_binary_mask(features):
    """
    Creates a random binary mask of a given dimension with half of its entries
    randomly set to 1s.

    :param features: Dimension of mask.
    :return: Binary mask with half of its entries set to 1s, of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    weights = torch.ones(features).float()
    num_samples = features // 2 if features % 2 == 0 else features // 2 + 1
    indices = torch.multinomial(input=weights, num_samples=num_samples, replacement=False)
    mask[indices] += 1
    return mask


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def cbrt(x):
    """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)


def get_temperature(max_value, bound=1 - 1e-3):
    """
    For a dataset with max value 'max_value', returns the temperature such that

        sigmoid(temperature * max_value) = bound.

    If temperature is greater than 1, returns 1.

    :param max_value:
    :param bound:
    :return:
    """
    max_value = torch.Tensor([max_value])
    bound = torch.Tensor([bound])
    temperature = min(-(1 / max_value) * (torch.log1p(-bound) - torch.log(bound)), 1)
    return temperature


def is_bool(x):
    return isinstance(x, bool)


def is_int(x):
    return isinstance(x, int)


def is_positive_int(x):
    return is_int(x) and x > 0


def is_nonnegative_int(x):
    return is_int(x) and x >= 0


def is_power_of_two(n):
    if is_positive_int(n):
        return not n & (n - 1)
    else:
        return False
