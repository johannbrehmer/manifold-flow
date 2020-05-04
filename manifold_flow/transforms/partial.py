""" Partial transformations """

import logging
import torch

from manifold_flow import transforms

logger = logging.getLogger(__name__)


class PartialTransform(transforms.Transform):
    """A base class for partial transformations, i.e. those where some features are transformed by some operation while others are passed along untransformed (somewhat similar
    to coupling layers, except that the identity features do not affect the transformation of the transformed features, and the transformation is not necessarily elementwise).
    """

    def __init__(self, mask, transform):
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError("Mask must be a 1-dim tensor.")
        if mask.numel() <= 0:
            raise ValueError("Mask can't be empty.")

        super().__init__()
        self.features = len(mask)
        features_vector = torch.arange(self.features)

        self.register_buffer("identity_features", features_vector.masked_select(mask <= 0))
        self.register_buffer("transform_features", features_vector.masked_select(mask > 0))

        assert self.num_identity_features + self.num_transform_features == self.features

        self.transform = transform

    @property
    def num_identity_features(self):
        return len(self.identity_features)

    @property
    def num_transform_features(self):
        return len(self.transform_features)

    def forward(self, inputs, context=None, full_jacobian=False):
        if inputs.shape[1] != self.features:
            raise ValueError("Expected features = {}, got {}.".format(self.features, inputs.shape[1]))

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        # logabsdet can be either log abs det or Jacobian, depending on full_jacobian
        transform_split, transform_logabsdet = self.transform(transform_split, context=context, full_jacobian=full_jacobian)

        if full_jacobian:
            batchsize = inputs.size(0)
            logabsdet = torch.zeros((batchsize,) + inputs.size()[1:] + inputs.size()[1:])
            logabsdet[:, self.identity_features, self.identity_features] = 1.0
            logabsdet[:, self.transform_features, :] = transform_logabsdet
        else:
            logabsdet = transform_logabsdet

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features, ...] = identity_split
        outputs[:, self.transform_features, ...] = transform_split

        return outputs, logabsdet

    def inverse(self, inputs, context=None, full_jacobian=False):
        if inputs.shape[1] != self.features:
            raise ValueError("Expected features = {}, got {}.".format(self.features, inputs.shape[1]))

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        # logabsdet can be either log abs det or Jacobian, depending on full_jacobian
        transform_split, transform_logabsdet = self.transform.inverse(transform_split, context=context, full_jacobian=full_jacobian)

        if full_jacobian:
            batchsize = inputs.size(0)
            logabsdet = torch.zeros((batchsize,) + inputs.size()[1:] + inputs.size()[1:])
            logabsdet[:, self.identity_features, self.identity_features] = 1.0
            logabsdet[:, self.transform_features, :] = transform_logabsdet
        else:
            logabsdet = transform_logabsdet

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features, ...] = identity_split
        outputs[:, self.transform_features, ...] = transform_split

        return outputs, logabsdet
