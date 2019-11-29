import torch
from torch import nn
import logging

from manifold_flow.utils.various import product
from manifold_flow.utils import vector_transforms, image_transforms
from manifold_flow import distributions, transforms

logger = logging.getLogger(__name__)


class Projection(transforms.Transform):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_dim_total = product(input_dim)
        self.output_dim_total = product(output_dim)
        self.mode_in = "vector" if isinstance(input_dim, int) else "image"
        self.mode_out = "vector" if isinstance(input_dim, int) else "image"

        logger.debug("Set up projection from %s with dimension %s to %s with dimension %s", self.mode_in, self.input_dim, self.mode_out, self.output_dim)

        assert self.input_dim_total >= self.output_dim_total, "Input dimension has to be larger than output dimension"

    def forward(self, inputs, **kwargs):
        if self.mode_in == "vector" and self.mode_out == "vector":
            u = inputs[:, : self.output_dim]
        elif self.mode_in == "image" and self.mode_out == "vector":
            u = inputs.view(inputs.size(0), -1)
            u = u[:, : self.output_dim]
        else:
            raise NotImplementedError("Unsuppoorted projection modes {}, {}".format(self.mode_in, self.mode_out))
        return u

    def inverse(self, inputs, **kwargs):
        if self.mode_in == "vector" and self.mode_out == "vector":
            x = torch.cat((inputs, torch.zeros(inputs.size(0), self.input_dim - self.output_dim)), dim=1)
        elif self.mode_in == "image" and self.mode_out == "vector":
            c, h, w = self.input_dim
            x = torch.cat((inputs, torch.zeros(inputs.size(0), self.input_dim_total - self.output_dim)), dim=1)
            x = x.view(inputs.size(0), c, h, w)
        else:
            raise NotImplementedError("Unsuppoorted projection modes {}, {}".format(self.mode_in, self.mode_out))
        return x


class ManifoldFlow(nn.Module):
    def __init__(
        self,
        data_dim,
        latent_dim=8,
        inner_transform="affine-coupling",
        outer_transform="affine-coupling",
        steps_inner=5,
        steps_outer=3,
        context_features=None,
        apply_context_to_outer=True,
        inner_transform_kwargs=None,
        outer_transform_kwargs=None,
    ):
        super(ManifoldFlow, self).__init__()

        assert latent_dim < data_dim

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.total_data_dim = product(data_dim)
        self.total_latent_dim = product(latent_dim)
        self.apply_context_to_outer = apply_context_to_outer

        self.latent_distribution = distributions.StandardNormal((self.total_latent_dim,))
        self.projection = Projection(self.total_data_dim, self.total_latent_dim)

        inner_transform_kwargs = {} if inner_transform_kwargs is None else inner_transform_kwargs
        outer_transform_kwargs = {} if outer_transform_kwargs is None else outer_transform_kwargs

        if isinstance(self.data_dim, int):
            if isinstance(outer_transform, str):
                logger.debug("Creating default outer transform for scalar data with base type %s", outer_transform)
                self.outer_transform = vector_transforms.create_transform(
                    data_dim, steps_outer, base_transform_type=outer_transform, context_features=context_features if apply_context_to_outer else None, **outer_transform_kwargs
                )
            else:
                self.outer_transform = outer_transform
        else:
            c, h, w = data_dim
            if isinstance(outer_transform, str):
                logger.debug("Creating default outer transform for image data")
                assert context_features is None
                self.outer_transform = image_transforms.create_transform(c, h, w, steps_outer)
            else:
                self.outer_transform = outer_transform

        if isinstance(inner_transform, str):
            logger.debug("Creating default inner transform with base type %s", outer_transform)
            self.inner_transform = vector_transforms.create_transform(latent_dim, steps_inner, base_transform_type=inner_transform, context_features=context_features, **inner_transform_kwargs)
        elif inner_transform is None:
            self.inner_transform = transforms.IdentityTransform()
        else:
            self.inner_transform = inner_transform

        self._report_model_parameters()

    def forward(self, x, context=None):
        # Encode
        u, h, log_det_inner, jacobian_outer = self._encode(x, context=context)

        # Decode
        x = self.decode(u, context=context)

        # Log prob
        log_prob = self._log_prob(u, log_det_inner, jacobian_outer)

        return x, log_prob, u

    def encode(self, x, context=None):
        u, _, _, _ = self._encode(x, context=context, calculate_jacobian=False)
        return u

    def decode(self, u, context=None):
        h, _ = self.inner_transform.inverse(u, context=context)
        h = self.projection.inverse(h)
        x, _ = self.outer_transform.inverse(h, context=context if self.apply_context_to_outer else None)
        return x

    def log_prob(self, x, context=None):
        # Encode
        u, _, log_det_inner, jacobian_outer = self._encode(x, context=context)

        # Log prob
        log_prob = self._log_prob(u, log_det_inner, jacobian_outer)

        return log_prob

    def sample(self, u=None, n=1, context=None):
        if u is None:
            u = self.latent_distribution.sample(n)
        x = self.decode(u, context=context)
        return x

    def _encode(self, x, calculate_jacobian=True, context=None):
        if calculate_jacobian:
            x.requires_grad = True
        h, jacobian_outer = self.outer_transform(x, full_jacobian=calculate_jacobian, context=context if self.apply_context_to_outer else None)
        h = self.projection(h)
        u, log_det_inner = self.inner_transform(h, context=context)
        if calculate_jacobian:
            return u, h, log_det_inner, jacobian_outer
        else:
            return u, h, None, None

    def _log_prob(self, u, log_det_inner, jacobian_outer):
        jacobian_outer = jacobian_outer[:, :, : self.latent_dim]
        jtj = torch.bmm(torch.transpose(jacobian_outer, -2, -1), jacobian_outer)
        log_det_outer = -0.5 * torch.slogdet(jtj)[1]

        log_prob = self.latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + log_det_outer + log_det_inner

        return log_prob

    def _report_model_parameters(self):
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size = all_params * (32 / 8)  # Bytes
        logger.debug("Created manifold flow with %.1f M parameters (%.1f M trainable) with an estimated size of %.1f GB", all_params / 1e6, trainable_params / 1.0e6, size / 1.0e9)
