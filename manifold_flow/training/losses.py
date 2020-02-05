import torch
from torch.nn import MSELoss
from geomloss import SamplesLoss


def nll(x_pred, x_true, log_p):
    return -torch.mean(log_p)


def mse(x_pred, x_true, log_p):
    return MSELoss()(x_pred, x_true)


def make_sinkhorn_divergence(blur=0.05, scaling=0.7, p=2, backend="auto"):
    """ See http://www.kernel-operations.io/geomloss/api/pytorch-api.html """
    sinkhorn = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=scaling, backend=backend)

    def sinkhorn_divergence(x_gen, x_true, log_p):
        return sinkhorn(x_gen, x_true)

    return sinkhorn_divergence


#
# def make_generalized_energy_distance(blur=0.05, scaling=0.5, p=2, backend="auto"):
#     """ See http://www.kernel-operations.io/geomloss/api/pytorch-api.html """
#     sinkhorn = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=scaling, backend=backend)
#
#     def generalized_energy_distance(x_gen, x_true, log_p):
#         batchsize = x_gen.size(0)
#         x_gen0 = x_gen[: batchsize // 2]
#         x_gen1 = x_gen[batchsize // 2 :]
#         x_true0 = x_true[: batchsize // 2]
#         x_true1 = x_true[batchsize // 2 :]
#
#         loss = (
#             sinkhorn(x_gen0, x_true0)
#             + sinkhorn(x_gen0, x_true1)
#             + sinkhorn(x_gen1, x_true0)
#             + sinkhorn(x_gen1, x_true1)
#             - 2.0 * sinkhorn(x_gen0, x_gen1)
#             - 2.0 * sinkhorn(x_true0, x_true1)
#         )
#         return loss
#
#     return generalized_energy_distance
#
#
# def make_conditional_generalized_energy_distance(blur=0.05, scaling=0.5, p=2, backend="auto"):
#     """ See http://www.kernel-operations.io/geomloss/api/pytorch-api.html """
#     sinkhorn = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=scaling, backend=backend)
#
#     def generalized_energy_distance(x_gen, x_true, log_p):
#         batchsize = x_gen.size(0)
#         x_gen0 = x_gen[: batchsize // 2]
#         x_gen1 = x_gen[batchsize // 2 :]
#         x_true0 = x_true[: batchsize // 2]
#         x_true1 = x_true[batchsize // 2 :]
#
#         loss = 2.0 * sinkhorn(x_gen0, x_true1) + 2.0 * sinkhorn(x_gen1, x_true0) - 2.0 * sinkhorn(x_gen0, x_gen1) - 2.0 * sinkhorn(x_true0, x_true1)
#         return loss
#
#     return generalized_energy_distance
