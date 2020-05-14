import torch
import numpy as np
import logging
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def save_model_after_every_epoch(filename):
    """ Saves model checkpoints. """

    def callback(i_epoch, model, loss_train, loss_val, subset=None, trainer=None, last_batch=None):
        if i_epoch in [0, 1, 4] or (i_epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), filename.format(i_epoch))

    return callback


def plot_sample_images(filename, context=None):
    """ Saves model checkpoints. """

    def callback(i_epoch, model, loss_train, loss_val, subset=None, trainer=None, last_batch=None):
        if i_epoch in [0, 1, 4] or (i_epoch + 1) % 10 == 0:
            x = model.sample(n=30, context=context).detach().cpu().numpy()
            x = np.clip(np.transpose(x, [0, 2, 3, 1]) / 256.0, 0.0, 1.0)

            plt.figure(figsize=(6 * 3.0, 5 * 3.0))
            for i in range(30):
                plt.subplot(5, 6, i + 1)
                plt.imshow(x[i])
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.savefig(filename.format(i_epoch))
            plt.close()

    return callback


def plot_reco_images(filename):
    """ Saves model checkpoints. """

    def callback(i_epoch, model, loss_train, loss_val, subset=None, trainer=None, last_batch=None):
        if last_batch is None:
            return

        if i_epoch == 0 or (i_epoch + 1) % 10 == 0:
            x = last_batch["x"]
            x_reco = last_batch["x_reco"]

            x = np.clip(np.transpose(x, [0, 2, 3, 1]) / 256.0, 0.0, 1.0)
            x_reco = np.clip(np.transpose(x_reco, [0, 2, 3, 1]) / 256.0, 0.0, 1.0)

            plt.figure(figsize=(6 * 3.0, 5 * 3.0))
            for i in range(15):
                plt.subplot(5, 6, 2 * i + 1)
                plt.imshow(x[i])
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)

                plt.subplot(5, 6, 2 * i + 2)
                plt.imshow(x_reco[i])
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)

            plt.tight_layout()
            plt.savefig(filename.format(i_epoch))
            plt.close()

    return callback


def print_mf_latent_statistics():
    """ Prints debug info about size of weights. """

    def callback(i_epoch, model, loss_train, loss_val, subset=None, trainer=None, last_batch=None):
        if last_batch is None:
            return

        u = last_batch["u"]

        logger.debug(f"           Latent variables: mean = {np.mean(u):>8.5f}")
        logger.debug(f"                             std  = {np.std(u):>8.5f}")
        logger.debug(f"                             min  = {np.min(u):>8.5f}")
        logger.debug(f"                             max  = {np.max(u):>8.5f}")

    return callback


def print_mf_weight_statistics():
    """ Prints debug info about size of weights. """

    def callback(i_epoch, model, loss_train, loss_val, subset=None, trainer=None, train_loader=None, val_loader=None):
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
