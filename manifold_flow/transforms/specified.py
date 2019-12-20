import torch
import logging
import numpy as np

from manifold_flow import transforms
from manifold_flow.utils.various import batch_jacobian

logger = logging.getLogger(__name__)


class SphericalCoordinates(transforms.Transform):
    """ Translates the first d+1 data components from Cartesian to the hyperspherical coordinates of a d-sphere and a radial coordinate, leaving the remaining variables
    invariant. Only supports  """

    def __init__(self, n, r0=0.):
        """ n is the dimension of the hyperspherical coordinates (as in n-sphere). A circle would be n = 1. r0 is subtracted from the radial coordinate, so with r = 1 deviations
        from the unit sphere are calculated. """

        super().__init__()

        self.n = n
        self.r0 = r0

    def forward(self, inputs, context=None, full_jacobian=False):
        assert len(inputs.size()) == 2, "Spherical coordinates only support 1-d data"

        outputs = self._cartesian_to_spherical(inputs)
        jacobian = batch_jacobian(outputs, inputs)

        if not full_jacobian:
            _, logdet = torch.slogdet(jacobian)
            return outputs, logdet

    def inverse(self, inputs, context=None, full_jacobian=False):
        assert len(inputs.size()) == 2, "Spherical coordinates only support 1-d data"

        outputs = self._spherical_to_cartesian(inputs)
        jacobian = batch_jacobian(outputs, inputs)

        if not full_jacobian:
            _, logdet = torch.slogdet(jacobian)
            return outputs, logdet

    def _split_spherical(self, spherical):
        batchsize = spherical.size(0)
        d = spherical.size(1)

        phi = spherical[:, :self.n]
        dr = spherical[:, self.n].view(batchsize, 1)
        others = spherical[:, self.n:]

        return (batchsize, d), (phi, dr, others)

    def _spherical_to_cartesian(self, inputs):
        (batchsize, d), (phi, dr, others) = self._split_spherical(inputs)
        r = dr + self.r0

        a1 = torch.cat((np.pi * torch.ones((batchsize, 1)), phi), dim=1)  # (batchsize, n+1)
        a0 = torch.cat((2 * np.pi * torch.ones((batchsize, 1)), phi), dim=1)  # (batchsize, n+1)

        sins = torch.sin(a1)  # (batchsize, n+1), first row is ones
        sins = torch.cumprod(sins, dim=1)
        coss = torch.cos(a0)  # (batchsize, n+1), first row is ones
        coss = torch.roll(coss, -1, dims=1)  # (batchsize, n+1), last row is ones

        unit_sphere = sins * coss
        sphere = unit_sphere * r[:, np.newaxis]
        outputs = torch.cat((sphere, others), dim=1)

        return outputs

    def _cartesian_to_spherical(self, inputs):
        # Calculate hyperspherical coordinates one by one
        phis = []
        for i in range(self.n):
            r_ = torch.sum(inputs[:, i : self.n + 1] ** 2, dim=1) ** 0.5
            phis.append(torch.arccos(inputs[:, i] / r_))

        # Special case for last component, see https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
        phis[-1] = torch.where(inputs[:, self.d] < 0.0, 2.0 * np.pi - phis[-1], phis[-1])

        # Radial coordinate
        r = torch.sum(inputs[:, : self.d + 1] ** 2, dim=1) ** 0.5
        dr = r - self.r0

        # Combine
        others = inputs[:,self.n:]
        outputs = torch.cat(phis + [dr, others], dim=1)

        return outputs
