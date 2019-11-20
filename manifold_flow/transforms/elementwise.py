import torch
import numpy as np
import logging

from manifold_flow import utils, transforms
from manifold_flow.transforms import splines


logger = logging.getLogger(__name__)


class ConditionalAffineScalarTransform(transforms.Transform):
    """Computes X = X * scale(context) + shift(context), where (scale, shift) are given by param_net(context).

    param_net takes as input the context with shape (batchsize, context_features) or None,
    its output has to have shape (batchsize, 2). """

    def __init__(self, param_net):
        super().__init__()

        self.param_net = param_net

    def get_scale_and_shift(self, context):
        scale_and_shift = self.param_net(context)
        scale = torch.exp(scale_and_shift[:, 0].unsqueeze(1))
        shift = scale_and_shift[:, 1].unsqueeze(1)

        num_dims = torch.prod(torch.tensor([1,]), dtype=torch.float)
        logabsdet = torch.log(torch.abs(scale.squeeze()) + 1.e-6) * num_dims
        return scale, shift, logabsdet

    def forward(self, inputs, context=None):
        scale_and_shift = self.param_net(context)
        scale = torch.exp(scale_and_shift[:, 0].unsqueeze(1))
        shift = scale_and_shift[:, 1].unsqueeze(1)

        # logger.debug("context: %s, scale: %s, shift: %s", context[0], scale[0], shift[0])

        num_dims = torch.prod(torch.tensor(inputs.shape[1:]), dtype=torch.float)
        outputs = inputs * scale + shift
        logabsdet = torch.log(torch.abs(scale.squeeze()) + 1.e-6) * num_dims

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        scale_and_shift = self.param_net(context)
        scale = torch.exp(scale_and_shift[:, 0].unsqueeze(1))
        shift = scale_and_shift[:, 1].unsqueeze(1)

        num_dims = torch.prod(torch.tensor(inputs.shape[1:]), dtype=torch.float)
        outputs = (inputs - shift) / scale
        logabsdet = - torch.log(torch.abs(scale) + 1.e-6) * num_dims

        return outputs, logabsdet


class ElementwisePiecewiseRationalQuadraticTransform(transforms.Transform):
    def __init__(self,
                 param_net=None,
                 num_bins=10,
                 tails="linear",
                 tail_bound=100.,
                 min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE,
                 ):

        super(ElementwisePiecewiseRationalQuadraticTransform, self).__init__()

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        if param_net is None:
            self.param_net = None
            self.params = torch.zeros(self._output_dim_multiplier())
            torch.nn.init.normal_(self.params)
            self.params = torch.nn.Parameter(self.params)
        else:
            self.param_net = param_net
            self.params = None

    def forward(self, inputs, context=None, full_jacobian=False):
        if self.param_net is not None:
            assert context is not None
            params = self.param_net(context)
        else:
            params = self.params
        outputs, logabsdet = self._elementwise_forward(inputs, params, full_jacobian=full_jacobian)
        return outputs, logabsdet

    def inverse(self, inputs, context=None, full_jacobian=False):
        params = self.param_net(context)
        outputs, logabsdet = self._elementwise_inverse(inputs, params, full_jacobian=full_jacobian)
        return outputs, logabsdet

    def _output_dim_multiplier(self):
        if self.tails == 'linear':
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _elementwise(self, inputs, params, inverse=False, full_jacobian=False):
        if full_jacobian:
            raise NotImplementedError

        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = params.view(
            batch_size,
            features,
            self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[...,:self.num_bins]
        unnormalized_heights = transform_params[...,self.num_bins:2*self.num_bins]
        unnormalized_derivatives = transform_params[...,2*self.num_bins:]

        if hasattr(self.param_net, 'hidden_features'):
            unnormalized_widths /= np.sqrt(self.param_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.param_net.hidden_features)

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == 'linear':
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {
                'tails': self.tails,
                'tail_bound': self.tail_bound
            }
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, utils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, params, full_jacobian=False):
        return self._elementwise(inputs, params, full_jacobian=full_jacobian)

    def _elementwise_inverse(self, inputs, params, full_jacobian=False):
        return self._elementwise(inputs, params, inverse=True, full_jacobian=full_jacobian)
