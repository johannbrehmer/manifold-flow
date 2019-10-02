#! /usr/bin/env python

import numpy as np
import logging
import sys
import torch
import argparse
from matplotlib import pyplot as plt
from torch import optim

sys.path.append("../")

from manifold_flow.flows.autoencoding_flow import TwoStepAutoencodingFlow
from manifold_flow.flows.flow import Flow
from manifold_flow.trainer import AutoencodingFlowTrainer, NumpyDataset
from manifold_flow.losses import nll, mse
from manifold_flow.utils import product, save_image
from aef_data.images import get_data, Preprocess

logger = logging.getLogger(__name__)


def train(
    model_filename,
    dataset="tth",
    data_dim=None,
    latent_dim=10,
    flow_steps_inner=5,
    flow_steps_outer=5,
    batch_size=128,
    epochs=(20, 20),
    alpha=1.0e-3,
    lr=(1.0e-4, 1.0e-6),
    base_dir=".",
):
    logger.info("Starting training of model %s on data set %s", model_filename, dataset)

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Data
    if dataset == "tth":
        x = np.load("{}/data/tth/x_train.npy".format(base_dir))
        x_means = np.mean(x, axis=0)
        x_stds = np.std(x, axis=0)
        x = (x - x_means[np.newaxis, :]) / x_stds[np.newaxis, :]
        y = np.ones(x.shape[0])
        data = NumpyDataset(x, y)
        data_dim = 48
        logger.info("Loaded tth data with %s dimensions", data_dim)

    elif dataset == "gaussian":
        assert data_dim is not None
        x = np.load(
            "{}/data/gaussian/gaussian_8_{}_x_train.npy".format(base_dir, data_dim)
        )
        y = np.ones(x.shape[0])
        data = NumpyDataset(x, y)
        logger.info("Loaded linear Gaussian data with %s dimensions", data_dim)

    elif dataset == "spherical_gaussian":
        assert data_dim is not None
        x = np.load(
            "{}/data/spherical_gaussian/spherical_gaussian_15_{}_x_train.npy".format(
                base_dir, data_dim
            )
        )
        y = np.ones(x.shape[0])
        data = NumpyDataset(x, y)
        logger.info("Loaded spherical Gaussian data with %s dimensions", data_dim)

    elif dataset == "cifar":
        data, data_dim = get_data("cifar-10", 8, base_dir + "/data/", train=True)
        logger.info("Loaded CIFAR data with dimensions %s", data_dim)

    elif dataset == "imagenet":
        data, data_dim = get_data(
            "imagenet-64-fast", 8, base_dir + "/data/", train=True
        )
        logger.info("Loaded ImageNet data with dimensions %s", data_dim)

    else:
        raise NotImplementedError("Unknown dataset {}".format(dataset))

    # Stop simulations where latent dim is larger than x dim
    if latent_dim is not None and product(latent_dim) > product(data_dim):
        logger.info("Latent dim is larger than data dim, skipping this")
        return

    # Model
    if latent_dim is None:
        logger.info("Creating plain flow")
        model = Flow(data_dim=data_dim, steps=flow_steps_outer)

    else:
        logger.info("Creating auto-encoding flow with %s latent dimensions", latent_dim)
        model = TwoStepAutoencodingFlow(
            data_dim=data_dim,
            latent_dim=latent_dim,
            steps_inner=flow_steps_inner,
            steps_outer=flow_steps_outer,
        )

    # Trainer
    trainer = AutoencodingFlowTrainer(model, double_precision=True)

    # Callbacks: save model after each epoch, and maybe generate a few images
    def save_model1(epoch, model, loss_train, loss_val):
        torch.save(
            model.state_dict(),
            "{}/data/flows/{}_phase1_epoch{}.pt".format(
                base_dir, model_filename, epoch
            ),
        )

    def save_model2(epoch, model, loss_train, loss_val):
        torch.save(
            model.state_dict(),
            "{}/data/flows/{}_phase2_epoch{}.pt".format(
                base_dir, model_filename, epoch
            ),
        )

    def sample1(epoch, model, loss_train, loss_val):
        if dataset not in ["cifar", "imagenet"]:
            return

        cols, rows = 4, 4

        model.eval()
        with torch.no_grad():
            preprocess = Preprocess(8)
            samples = model.sample(n=cols*rows)
            samples = preprocess.inverse(samples)
            samples = samples.cpu()
        samples = samples.detach().numpy()
        samples = np.moveaxis(samples, 1, -1)

        plt.figure(figsize=(cols*4, rows*4))
        for i, x in enumerate(samples):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(x)
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(
            "{}/figures/training/images_{}_phase1_{}.pdf".format(
                base_dir, model_filename, epoch
            )
        )
        plt.close()

    def sample2(epoch, model, loss_train, loss_val):
        if dataset not in ["cifar", "imagenet"]:
            return

        cols, rows = 4, 4

        model.eval()
        with torch.no_grad():
            preprocess = Preprocess(8)
            samples = model.sample(n=cols*rows)
            samples = preprocess.inverse(samples)
            samples = samples.cpu()
        samples = samples.detach().numpy()
        samples = np.moveaxis(samples, 1, -1)

        plt.figure(figsize=(cols*4, rows*4))
        for i, x in enumerate(samples):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(x)
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(
            "{}/figures/training/images_{}_phase2_{}.pdf".format(
                base_dir, model_filename, epoch
            )
        )
        plt.close()

    # Train
    if latent_dim is None:
        logger.info("Starting training on NLL")
        trainer.train(
            optimizer=torch.optim.Adam,
            dataset=data,
            loss_functions=[nll],
            loss_labels=["NLL"],
            loss_weights=[1.0],
            batch_size=batch_size,
            epochs=epochs,
            verbose="all",
            initial_lr=lr[0],
            scheduler=optim.lr_scheduler.CosineAnnealingLR,
            scheduler_kwargs={"eta_min": lr[1],},
            callbacks=[save_model1, sample1],
        )
    else:
        logger.info("Starting training on MSE")
        trainer.train(
            optimizer=torch.optim.Adam,
            dataset=data,
            loss_functions=[mse],
            loss_labels=["MSE"],
            loss_weights=[1.0],
            batch_size=batch_size,
            epochs=epochs // 2,
            verbose="all",
            initial_lr=lr[0],
            scheduler=optim.lr_scheduler.CosineAnnealingLR,
            scheduler_kwargs={"eta_min": lr[1],},
            callbacks=[save_model1, sample1],
        )
        logger.info("Starting training on MSE and NLL")
        trainer.train(
            optimizer=torch.optim.Adam,
            dataset=data,
            loss_functions=[mse, nll],
            loss_labels=["MSE", "NLL"],
            loss_weights=[1.0, alpha],
            batch_size=batch_size,
            epochs=epochs - epochs // 2,
            verbose="all",
            initial_lr=lr[0],
            scheduler=optim.lr_scheduler.CosineAnnealingLR,
            scheduler_kwargs={"eta_min": lr[1],},
            parameters=model.outer_transform.parameters(),
            callbacks=[save_model2, sample2],
        )

    # Save
    logger.info(
        "Saving model to %s", "{}/data/flows/{}.pt".format(base_dir, model_filename)
    )
    torch.save(
        model.state_dict(), "{}/data/flows/{}.pt".format(base_dir, model_filename)
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strong lensing experiments: simulation"
    )
    parser.add_argument("name", type=str, help="Model name.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tth",
        choices=["cifar", "imagenet", "tth", "gaussian", "spherical_gaussian"],
    )
    parser.add_argument("-x", type=int, default=None)
    parser.add_argument("--latent", type=int, default=None)
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--outer", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--lrdecay", type=float, default=1.e-3)
    parser.add_argument(
        "--dir",
        type=str,
        default="/Users/johannbrehmer/work/projects/ae_flow/autoencoded-flow",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
        datefmt="%H:%M",
        level=logging.DEBUG if args.debug else logging.INFO,
    )
    logger.info("Hi!")

    train(
        model_filename=args.name,
        dataset=args.dataset,
        data_dim=args.x,
        latent_dim=args.latent,
        flow_steps_inner=args.inner,
        flow_steps_outer=args.outer,
        batch_size=args.batchsize,
        epochs=args.epochs,
        alpha=args.alpha,
        lr=(args.lr, args.lr * args.lrdecay),
        base_dir=args.dir,
    )

    logger.info("All done! Have a nice day!")
