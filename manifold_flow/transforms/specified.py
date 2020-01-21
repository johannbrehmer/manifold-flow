import torch
import logging
import numpy as np

from manifold_flow import transforms
from manifold_flow.utils.various import batch_jacobian

logger = logging.getLogger(__name__)


class SphericalCoordinates(transforms.Transform):
    """ Translates the first d+1 data components from Cartesian to the hyperspherical coordinates of a d-sphere and a radial coordinate, leaving the remaining variables
    invariant. Spherical angles get linearly rescaled to (-1, 1).  """

    def __init__(self, n, r0=1.0):
        """ n is the dimension of the hyperspherical coordinates (as in n-sphere). A circle would be n = 1. r0 is subtracted from the radial coordinate, so with r = 1 deviations
        from the unit sphere are calculated. """

        super().__init__()

        self.n = n
        self.r0 = r0
        self._mask = None

    def forward(self, inputs, context=None, full_jacobian=False):
        """ Transforms Cartesian to hyperspherical coordinates """

        assert len(inputs.size()) == 2, "Spherical coordinates only support 1-d data"

        # Avoid NaNs by moving points slightly off the equator
        # inputs[:, self.n] = torch.where(inputs[:,self.n]**2 < 1.e-4, 1.e-2*torch.sign(inputs[:,self.n])*torch.ones_like(inputs[:,self.n]), inputs[:, self.n])
        mask = torch.zeros_like(inputs)
        mask[:, self.n] = 1.0
        mask = mask * (inputs ** 2 < 1.0e-4)
        replace = 1.0e-2 * torch.sign(inputs)
        inputs = mask * replace + (1.0 - mask) * inputs

        if not inputs.requires_grad:
            inputs.requires_grad = True

        outputs = self._cartesian_to_spherical(inputs)

        if full_jacobian:  # Should not be necessary
            jacobian = batch_jacobian(outputs, inputs)

            if torch.isnan(jacobian).any():
                for i in range(jacobian.size(0)):
                    if torch.isnan(jacobian[i]).any():
                        logger.warning("Spherical Jacobian contains NaNs")
                        logger.warning("  Cartesian: %s", inputs[i])
                        logger.warning("  Spherical: %s", outputs[i])
                        logger.warning("  Jacobian:  %s", jacobian[i])
                        raise RuntimeError

            return outputs, jacobian

        else:
            logdet = self._logdet(outputs, inverse=False)

            # Cross-check
            # jacobian = batch_jacobian(outputs, inputs)
            # logdet_check = torch.slogdet(jacobian)[1]

            return outputs, logdet

    def inverse(self, inputs, context=None, full_jacobian=False):
        """ Transforms hyperspherical to Cartesian coordinates """

        assert len(inputs.size()) == 2, "Spherical coordinates only support 1-d data"
        # if not inputs.requires_grad:
        #     inputs.requires_grad = True

        outputs, jacobian = self._spherical_to_cartesian(inputs, calculate_jacobian=full_jacobian)

        if full_jacobian:
            # check_jacobian = batch_jacobian(outputs, inputs)

            # old = check_jacobian.detach().numpy()[0]
            # new = jacobian.detach().numpy()[0]
            #
            # diff = new - old
            #
            # if torch.isnan(jacobian).any():
            #     for i in range(jacobian.size(0)):
            #         if torch.isnan(jacobian[i]).any():
            #             logger.warning("Spherical inverse Jacobian contains NaNs")
            #             logger.warning("  Spherical: %s", inputs[i])
            #             logger.warning("  Cartesian: %s", outputs[i])
            #             logger.warning("  Jacobian:  %s", jacobian[i])
            #             raise RuntimeError

            return outputs, jacobian

        else:
            logdet = self._logdet(inputs, inverse=True)

            # Cross-check
            # jacobian = batch_jacobian(outputs, inputs)
            # logdet_check = torch.slogdet(jacobian)[1]

            return outputs, logdet

    def _split_spherical(self, spherical):
        batchsize = spherical.size(0)
        d = spherical.size(1)

        phi = spherical[:, : self.n]
        dr = spherical[:, self.n].view(batchsize, 1)
        others = spherical[:, self.n + 1 :]

        return (batchsize, d), (phi, dr, others)

    def _spherical_to_cartesian(self, inputs, calculate_jacobian=False):
        (batchsize, d), (phi, dr, others) = self._split_spherical(inputs)
        r = dr + self.r0

        # Rescale phi
        first = (phi[:, :-1] + 1.0) * 0.5 * np.pi
        last = (phi[:, -1].unsqueeze(1) + 1.0) * np.pi
        phi = torch.cat((first, last), dim=1)

        a1 = torch.cat((0.5 * np.pi * torch.ones((batchsize, 1)), phi), dim=1)  # (batchsize, n+1), first row is pi, rest are angles
        a0 = torch.cat((2 * np.pi * torch.ones((batchsize, 1)), phi), dim=1)  # (batchsize, n+1), first row are 2pi, rest are angles

        sins = torch.sin(a1)  # (batchsize, n+1): 1, sin(phi0), sin(phi1), ...
        sin_prods = torch.cumprod(sins, dim=1)  # (batchsize, n+1): 1, sin(phi0), sin(phi0) sin(phi1), ...
        coss = torch.cos(a0)  # (batchsize, n+1): 1, cos(phi0), cos(phi1), ...
        coss_roll = torch.roll(coss, -1, dims=1)  # (batchsize, n+1): cos(phi0), cos(phi1), ..., 1

        x_unit_sphere = sin_prods * coss_roll
        x_sphere = x_unit_sphere * r.view((-1, 1))

        outputs = torch.cat((x_sphere, others), dim=1)

        # Jacobian calculation
        jacobian = None
        if calculate_jacobian:
            # dx / dphi
            cots = coss / sins  # (batchsize, n + 1): 1, cot(phi0), cot(phi1), ...
            cots_roll = torch.roll(cots, -1, dims=1)  # (batchsize, n + 1): cot(phi0), cot(phi1), ..., 1
            sin_prods_roll = torch.roll(sin_prods, -1, dims=1)  # (batchsize, n + 1): sin(phi0), sin(phi0)sin(phi1), ..., 1

            offdiags = torch.einsum("b,bi,bj->bij", r.squeeze(), x_sphere, cots_roll)  # (batchsize, n + 1, n + 1) where dim 1 -> x, dim 2 -> phi
            diags = torch.diag_embed(-1.0 * r * sin_prods_roll)  # (batchsize, n + 1, n + 1)

            filter = torch.eye(self.n + 1).unsqueeze(0)
            jac_phi = (1.0 - filter) * offdiags + filter * diags
            jac_phi = jac_phi[:, :, : self.n]  # (batchsize, n + 1, n)

            if self._mask is None or self._mask.size(0) != batchsize:
                # Make a matrix with shape (batchsize, n, n+1) that has on/below the diagonal of each (i, :, :n) part zeros elsewhere
                self._mask = torch.tril(torch.ones((self.n, self.n)))
                self._mask = torch.cat((self._mask, torch.ones((1, self.n))), dim=0)  # (n + 1, n)
                self._mask = torch.zeros((batchsize, self.n + 1, self.n)) + self._mask.unsqueeze(0)  # (batchsize, n + 1, n)
            jac_phi = torch.zeros((batchsize, self.n + 1, self.n)) + self._mask * jac_phi  # (batchsize, n + 1, n)

            # Effect of rescaling
            jac_phi = torch.cat([0.5 * np.pi * jac_phi[:, :, :-1], np.pi * jac_phi[:, :, -1].unsqueeze(2)], dim=2)

            # dx / dr
            jac_r = torch.zeros((batchsize, self.n + 1, 1)) + x_unit_sphere.unsqueeze(2)  # (batchsize, n + 1, 1)

            # dx / dothers
            jac_rest = torch.zeros((batchsize, d - self.n - 1, d - self.n - 1)) + torch.eye(d - self.n - 1).unsqueeze(0)  # (batchsize, d - (n+1), d - (n+1))

            # Combine
            jac_upper = torch.cat([jac_phi, jac_r, torch.zeros((batchsize, self.n + 1, d - self.n - 1))], dim=2)  # (batchsize, n+1, d)
            jac_lower = torch.cat([torch.zeros((batchsize, d - self.n - 1, self.n + 1)), jac_rest], dim=2)  # (batchsize, d - (n+1), d)
            jacobian = torch.cat([jac_upper, jac_lower], dim=1)  # (batchsize, d, d)

        return outputs, jacobian

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

        # Rescale
        phis = [-1.0 + 2.0 * phi / np.pi if i < self.n - 1 else -1.0 + phi / np.pi for i, phi in enumerate(phis)]

        # Radial coordinate
        r = torch.sum(inputs[:, : self.n + 1] ** 2, dim=1) ** 0.5
        dr = r - self.r0
        dr = dr.view((-1, 1))

        # Combine
        others = inputs[:, self.n + 1 :]
        outputs = torch.cat(phis + [dr, others], dim=1)

        return outputs

    def _logdet(self, spherical, inverse=False):
        (batchsize, d), (phi, dr, others) = self._split_spherical(spherical)
        r = dr + self.r0

        logdet = self.n * torch.log(r).squeeze()
        for i, phi_ in enumerate(torch.t(phi)):
            logdet = logdet + (self.n - i - 1) * torch.log(torch.abs(torch.sin(phi_)))

        if not inverse:
            logdet = -logdet

        # Rescaling
        if inverse:
            logdet = logdet + np.log(np.pi) + (self.n - 1) * np.log(0.5 * np.pi)
        else:
            logdet = logdet - np.log(np.pi) - (self.n - 1) * np.log(0.5 * np.pi)

        return logdet
