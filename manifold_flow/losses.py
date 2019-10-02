import torch
from torch.nn import MSELoss


def nll(x_pred, x_true, log_p):
    return -torch.mean(log_p)


def mse(x_pred, x_true, log_p):
    return MSELoss()(x_pred, x_true)
