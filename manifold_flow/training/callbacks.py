import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def save_model_after_every_epoch(filename):
    def callback(i_epoch, model, loss_train, loss_val):
        torch.save(model.state_dict(), filename.format(i_epoch))

    return callback


def print_mf_weight_statistics():
    def callback(i_epoch, model, loss_train, loss_val):
        try:
            models = [model.outer_transform, model.inner_transform]
            labels = ["outer transform weights:", "inner transform weights:"]
        except:
            models = [model.transform]
            labels = ["transform weights:"]

        for model_, label_ in zip(models, labels):
            weights = np.hstack([param.detach().numpy().flatten() for param in model_.parameters()])
            logger.debug(
                "           {:26.26s} mean {:>8.5f}, std {:>8.5f}, range {:>8.5f} ... {:>8.5f}".format(
                    label_, np.mean(weights), np.std(weights), np.min(weights), np.max(weights)
                )
            )

    return callback
