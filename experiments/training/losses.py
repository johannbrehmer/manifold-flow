import torch
from torch.nn import MSELoss, SmoothL1Loss
import logging
SamplesLoss = None

logger = logging.getLogger(__name__)


def nll(x_pred, x_true, log_p, t_pred=None, t_xz=None):
    """ Negative log likelihood """

    return -torch.mean(log_p)


def mse(x_pred, x_true, log_p, t_pred=None, t_xz=None):
    """ Reconstruction error """

    return MSELoss()(x_pred, x_true)


def smooth_l1_loss(x_pred, x_true, log_p, t_pred=None, t_xz=None):
    """ Reconstruction error """

    return SmoothL1Loss()(x_pred, x_true)


def score_mse(x_pred, x_true, log_p, t_pred=None, t_xz=None):
    """ SCANDAL score MSE """

    return MSELoss()(t_pred, t_xz)


def make_sinkhorn_divergence(blur=0.05, scaling=0.7, p=2, backend="auto"):
    """
    Creates Sinkhorn divergence loss.

    See http://www.kernel-operations.io/geomloss/api/pytorch-api.html
    """

    from geomloss import SamplesLoss  # Workaround for some of the clusters. TODO: make this prettier

    sinkhorn = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=scaling, backend=backend)

    def sinkhorn_divergence(x_gen, x_true, log_p, t_pred=None, t_xz=None):
        return sinkhorn(x_gen, x_true)

    return sinkhorn_divergence
