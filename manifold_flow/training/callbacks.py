import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def save_model_after_every_epoch(filename):
    """ Saves model checkpoints. """

    def callback(i_epoch, model, loss_train, loss_val, subset=None, trainer=None):
        torch.save(model.state_dict(), filename.format(i_epoch))

    return callback


def print_mf_weight_statistics():
    """ Prints debug info about size of weights. """

    def callback(i_epoch, model, loss_train, loss_val, subset=None, trainer=None):
        try:
            models = [model.outer_transform, model.inner_transform]
            labels = ["outer transform weights:", "inner transform weights:"]
        except:
            models = [model.transform]
            labels = ["transform weights:"]

        subset_str = "          " if subset is None or trainer is None else "  {:>2d} / {:>2d}:".format(subset, trainer)

        for model_, label_ in zip(models, labels):
            weights = np.hstack([param.detach().numpy().flatten() for param in model_.parameters()])
            logger.debug(
                "{} {:26.26s} mean {:>8.5f}, std {:>8.5f}, range {:>8.5f} ... {:>8.5f}".format(
                    subset_str, label_, np.mean(weights), np.std(weights), np.min(weights), np.max(weights)
                )
            )

    return callback
