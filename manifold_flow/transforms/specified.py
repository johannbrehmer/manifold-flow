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

        # Avoid NaNs by ignoring a pole
        inputs_[:, self.n+1] = torch.where(inputs[:,self.n+1] < 1.e-3, torch.zeros_like(inputs[:,self.n+1]))

        if not inputs.requires_grad:
            inputs.requires_grad = True

        outputs = self._cartesian_to_spherical(inputs)
        jacobian = batch_jacobian(outputs, inputs)

        if torch.isnan(jacobian).any():
            for i in range(jacobian.size(0)):
                if torch.isnan(jacobian[i]).any():
                    logger.warning("Spherical Jacobian contains NaNs")
                    logger.warning("  Cartesian: %s", inputs[i])
                    logger.warning("  Spherical: %s", outputs[i])
                    logger.warning("  Jacobian:  %s", jacobian[i])

                    self._spherical_to_cartesian(inputs[i:i+1])

                    raise RuntimeError

        if not full_jacobian:
            _, logdet = torch.slogdet(jacobian)
            return outputs, logdet

        return outputs, jacobian

    def inverse(self, inputs, context=None, full_jacobian=False):
        assert len(inputs.size()) == 2, "Spherical coordinates only support 1-d data"
        if not inputs.requires_grad:
            inputs.requires_grad = True

        outputs = self._spherical_to_cartesian(inputs)
        jacobian = batch_jacobian(outputs, inputs)

        if torch.isnan(jacobian).any():
            for i in range(jacobian.size(0)):
                if torch.isnan(jacobian[i]).any():
                    logger.warning("Spherical inverse Jacobian contains NaNs")
                    logger.warning("  Spherical: %s", inputs[i])
                    logger.warning("  Cartesian: %s", outputs[i])
                    logger.warning("  Jacobian:  %s", jacobian[i])

                    self._spherical_to_cartesian(inputs[i:i+1])

                    raise RuntimeError

        if not full_jacobian:
            _, logdet = torch.slogdet(jacobian)
            if torch.isnan(logdet).any():
                logger.warning("log det Jacobian contains NaNs: %s", logdet)
            return outputs, logdet

        return outputs, jacobian

    def _split_spherical(self, spherical):
        batchsize = spherical.size(0)
        d = spherical.size(1)

        phi = spherical[:, :self.n]
        dr = spherical[:, self.n].view(batchsize, 1)
        others = spherical[:, self.n + 1:]

        return (batchsize, d), (phi, dr, others)

    def _spherical_to_cartesian(self, inputs):
        (batchsize, d), (phi, dr, others) = self._split_spherical(inputs)
        r = dr + self.r0

        a1 = torch.cat((0.5 * np.pi * torch.ones((batchsize, 1)), phi), dim=1)  # (batchsize, n+1), first row is pi, rest are angles
        a0 = torch.cat((2 * np.pi * torch.ones((batchsize, 1)), phi), dim=1)  # (batchsize, n+1), first row are 2pi, rest are angles

        sins = torch.sin(a1)  # (batchsize, n+1), first row is ones, others are sin(angles)
        sins = torch.cumprod(sins, dim=1)  # (batchsize, n+1)
        coss = torch.cos(a0)  # (batchsize, n+1), first row is ones, others are sin(angles)
        coss = torch.roll(coss, -1, dims=1)  # (batchsize, n+1), last row is ones

        unit_sphere = sins * coss
        sphere = unit_sphere * r.view((-1, 1))

        outputs = torch.cat((sphere, others), dim=1)

        return outputs

    def _cartesian_to_spherical(self, inputs):
        # Calculate hyperspherical coordinates one by one
        phis = []
        for i in range(self.n):
            r_ = torch.sum(inputs[:, i : self.n + 1] ** 2, dim=1) ** 0.5
            phi_ = torch.acos(inputs[:, i] / r_)

            # The cartesian -> spherical transformation is not unique when inputs_i to inputs_n are all zero
            # In that case we can choose to set the coordinate to 0
            # This choice maybe avoids derivatives evaluating to NaNs? Noo... :/
            # phi_ = torch.where(r_ < 0.04, torch.zeros_like(phi_), phi_)

            # Actually, we have to be more aggressive to save the gradients from becoming NaNs!
            # When inputs_(i+1) to inputs_n are all zero, the argument to the arccos is very small,
            # either below zero (when inputs_i is negative) or above (when inputs_i is positive).
            # In this case we can fix this angle to be zero or pi.
            # But it also doesn't suffice...
            # phi_ = torch.where(
            #     torch.sum(inputs[:, i+1 : self.n + 1] ** 2, dim=1) < 0.000001,
            #     torch.where(
            #         inputs[:,i] < 0.,
            #         np.pi * torch.ones_like(phi_),
            #         torch.zeros_like(phi_)
            #     ),
            #     phi_
            # )
            # logger.debug(torch.sum(torch.sum(inputs[:, i+1 : self.n + 1] ** 2, dim=1) < 0.000001).item())

            phi_ = phi_.view((-1, 1))
            phis.append(phi_)

        # Special case for last component, see https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
        phis[-1] = torch.where(inputs[:, self.n] < 0.0, 2.0 * np.pi - phis[-1][:, 0], phis[-1][:, 0]).view((-1, 1))

        # Radial coordinate
        r = torch.sum(inputs[:, : self.n + 1] ** 2, dim=1) ** 0.5
        dr = r - self.r0
        dr = dr.view((-1, 1))

        # Combine
        others = inputs[:,self.n + 1:]
        outputs = torch.cat(phis + [dr, others], dim=1)

        return outputs
