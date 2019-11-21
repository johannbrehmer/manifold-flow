#! /usr/bin/env python

import numpy as np
import logging
import sys
import torch
import argparse
import os
from torch import optim

sys.path.append("../")

from manifold_flow.flows import ManifoldFlow, Flow, PIE
from manifold_flow.training import ManifoldFlowTrainer, losses, NumpyDataset

logger = logging.getLogger(__name__)


# def generate_callbacks(args):
#     # Callbacks: save model after each epoch, and maybe generate a few images
#     def save_model1(epoch, model, loss_train, loss_val):
#         torch.save(
#             model.state_dict(),
#             "{}/data/flows/{}_phase1_epoch{}.pt".format(
#                 base_dir, model_filename, epoch
#             ),
#         )
#
#     def save_model2(epoch, model, loss_train, loss_val):
#         torch.save(
#             model.state_dict(),
#             "{}/data/flows/{}_phase2_epoch{}.pt".format(
#                 base_dir, model_filename, epoch
#             ),
#         )
#
#     def sample1(epoch, model, loss_train, loss_val):
#         if dataset not in ["cifar", "imagenet"]:
#             return
#
#         cols, rows = 4, 4
#
#         model.eval()
#         with torch.no_grad():
#             preprocess = Preprocess(8)
#             samples = model.sample(n=cols*rows)
#             samples = preprocess.inverse(samples)
#             samples = samples.cpu()
#         samples = samples.detach().numpy()
#         samples = np.moveaxis(samples, 1, -1)
#
#         plt.figure(figsize=(cols*4, rows*4))
#         for i, x in enumerate(samples):
#             plt.subplot(rows, cols, i + 1)
#             plt.imshow(x)
#             plt.gca().get_xaxis().set_visible(False)
#             plt.gca().get_yaxis().set_visible(False)
#         plt.tight_layout()
#         plt.savefig(
#             "{}/figures/training/images_{}_phase1_{}.pdf".format(
#                 base_dir, model_filename, epoch
#             )
#         )
#         plt.close()
#
#     def sample2(epoch, model, loss_train, loss_val):
#         if dataset not in ["cifar", "imagenet"]:
#             return
#
#         cols, rows = 4, 4
#
#         model.eval()
#         with torch.no_grad():
#             preprocess = Preprocess(8)
#             samples = model.sample(n=cols*rows)
#             samples = preprocess.inverse(samples)
#             samples = samples.cpu()
#         samples = samples.detach().numpy()
#         samples = np.moveaxis(samples, 1, -1)
#
#         plt.figure(figsize=(cols*4, rows*4))
#         for i, x in enumerate(samples):
#             plt.subplot(rows, cols, i + 1)
#             plt.imshow(x)
#             plt.gca().get_xaxis().set_visible(False)
#             plt.gca().get_yaxis().set_visible(False)
#         plt.tight_layout()
#         plt.savefig(
#             "{}/figures/training/images_{}_phase2_{}.pdf".format(
#                 base_dir, model_filename, epoch
#             )
#         )
#         plt.close()


def train(args):
    if args.modelname is None:
        args.modelname = "{}_{}_{}_{}_{}_{}".format(
            args.algorithm, args.modellatentdim, args.dataset, args.truelatentdim, args.datadim, args.epsilon
        )
    logger.info("Training model %s algorithm %s and %s latent dims on data set %s (data dim %s, true latent dim %s)", args.modelname, args.algorithm, args.modellatentdim, args.dataset, args.datadim, args.truelatentdim)

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Data
    if args.dataset == "spherical_gaussian":
        x = np.load(
            "{}/data/spherical_gaussian/spherical_gaussian_{}_{}_{}_x_train.npy".format(
                args.dir, args.latentdim, args.datadim, args.epsilon
            )
        )
        y = np.ones(x.shape[0])
        dataset = NumpyDataset(x, y)
        logger.info("Loaded spherical Gaussian data")
    else:
        raise NotImplementedError("Unknown dataset {}".format(args.dataset))

    # Model
    if args.algorithm == "flow":
        logger.info("Creating standard flow with %s layers", args.outerlayers)
        model = Flow(data_dim=args.datadim, transform=args.outerlayers)
    elif args.algorithm == "pie":
        logger.info("Creating PIE with %s latent dimensions and %s + %s layers", args.modellatentdim, args.outerlayers, args.innerlayers)
        model = PIE(data_dim=args.datadim, latent_dim=args.modellatentdim, inner_transform=args.innerlayers, outer_transform=args.outerlayers)
    elif args.algorithm == "mf":
        logger.info("Creating manifold flow with %s latent dimensions and %s + %s layers", args.modellatentdim, args.outerlayers, args.innerlayers)
        model = ManifoldFlow(data_dim=args.datadim, latent_dim=args.modellatentdim, inner_transform=args.innerlayers, outer_transform=args.outerlayers)
    else:
        raise NotImplementedError("Unknown algorithm {}".format(args.algorithm))

    # Trainer
    trainer = ManifoldFlowTrainer(model)

    # Train
    if args.algorithm in ["flow", "pie"]:
        logger.info("Starting training on NLL")
        trainer.train(
            dataset=dataset,
            loss_functions=[losses.nll],
            loss_labels=["NLL"],
            loss_weights=[1.0],
            batch_size=args.batchsize,
            epochs=args.epochs,
            initial_lr=args.lr,
            scheduler=optim.lr_scheduler.CosineAnnealingLR,
        )
    else:
        logger.info("Starting training on MSE")
        trainer.train(
            dataset=dataset,
            loss_functions=[losses.mse, losses.nll],
            loss_labels=["MSE", "NLL"],
            loss_weights=[args.mfalpha, args.mfbeta],
            batch_size=args.batchsize,
            epochs=args.epochs // 2,
            initial_lr=args.lr,
            scheduler=optim.lr_scheduler.CosineAnnealingLR,
        )
        logger.info("Starting training on NLL")
        trainer.train(
            dataset=dataset,
            loss_functions=[losses.mse, losses.nll],
            loss_labels=["MSE", "NLL"],
            loss_weights=[0., 1.],
            batch_size=args.batchsize,
            epochs=args.epochs // 2,
            initial_lr=args.lr,
            scheduler=optim.lr_scheduler.CosineAnnealingLR,
        )

    # Save
    logger.info("Saving model to %s", "{}/experiments/models/{}.pt".format(args.dir, args.modelname))
    os.makedirs("{}/experiments/models".format(args.dir, args.modelname), exist_ok=True)
    torch.save(model.state_dict(), "{}/experiments/models/{}.pt".format(args.dir, args.modelname))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default=None, help="Model name.")
    parser.add_argument("--algorithm", type=str, default="mf", choices=["flow", "pie", "mf"])
    parser.add_argument("--dataset", type=str, default="spherical_gaussian", choices=["spherical_gaussian"],)

    parser.add_argument("--truelatentdim", type=int, default=9)
    parser.add_argument("--datadim", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.01)

    parser.add_argument("--modellatentdim", type=int, default=10)
    parser.add_argument("--outerlayers", type=int, default=5)
    parser.add_argument("--innerlayers", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.e-3)
    parser.add_argument("--alpha", type=float, default=100.)
    parser.add_argument("--beta", type=float, default=1.e-3)

    parser.add_argument("--dir", type=str, default="../")
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
    train(args)
    logger.info("All done! Have a nice day!")
