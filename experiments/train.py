#! /usr/bin/env python

""" Top-level script for training models """

import numpy as np
import logging
import sys
import torch
import configargparse
import copy
from torch import optim

sys.path.append("../")

from training import (
    ForwardTrainer,
    losses,
    ConditionalForwardTrainer,
    callbacks,
    AdversarialTrainer,
    ConditionalAdversarialTrainer,
    AlternatingTrainer,
    # VarDimForwardTrainer,
    # ConditionalVarDimForwardTrainer,
    SCANDALForwardTrainer,
)
from datasets import load_simulator, load_training_dataset, SIMULATORS
from utils import create_filename, create_modelname
from architectures import create_model
from architectures.create_model import ALGORITHMS

logger = logging.getLogger(__name__)


def parse_args():
    """ Parses command line arguments for the training """

    parser = configargparse.ArgumentParser()

    # What what what
    parser.add_argument("--modelname", type=str, default=None, help="Model name. Algorithm, latent dimension, dataset, and run are prefixed automatically.")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="flow",
        choices=ALGORITHMS,
        help="Algorithm: flow (for AF), mf (for FOM, MFMF), emf (for MFMFE), pie (for PIE), gamf (for MFMF-OT)...",
    )
    parser.add_argument(
        "--dataset", type=str, default="spherical_gaussian", choices=SIMULATORS, help="Dataset: spherical_gaussian, power, lhc, lhc40d, lhc2d, and some others"
    )
    parser.add_argument("-i", type=int, default=0, help="Run number")

    # Dataset details
    parser.add_argument("--truelatentdim", type=int, default=2, help="True manifold dimensionality (for datasets where that is variable)")
    parser.add_argument("--datadim", type=int, default=3, help="True data dimensionality (for datasets where that is variable)")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Noise term (for datasets where that is variable)")

    # Model details
    parser.add_argument("--modellatentdim", type=int, default=2, help="Model manifold dimensionality")
    parser.add_argument("--specified", action="store_true", help="Prescribe manifold chart: FOM instead of MFMF")
    parser.add_argument(
        "--outertransform",
        type=str,
        default="rq-coupling",
        help="Type of transformations for f: affine-coupling, quadratic-coupling, rq-coupling, affine-autoregressive, quadratic-autoregressive, rq-autoregressive. See Neural Spline Flow paper.",
    )
    parser.add_argument(
        "--innertransform",
        type=str,
        default="rq-coupling",
        help="Type of transformations for h: affine-coupling, quadratic-coupling, rq-coupling, affine-autoregressive, quadratic-autoregressive, rq-autoregressive. See Neural Spline Flow paper.",
    )
    parser.add_argument(
        "--lineartransform",
        type=str,
        default="permutation",
        help="Type of linear transformations inserted between the base transformations: linear, permutation. See Neural Spline Flow paper.",
    )
    parser.add_argument("--outerlayers", type=int, default=5, help="Number of transformations in f (not counting linear transformations)")
    parser.add_argument("--innerlayers", type=int, default=5, help="Number of transformations in h (not counting linear transformations)")
    parser.add_argument(
        "--conditionalouter", action="store_true", help="If dataset is conditional, use this to make f conditional (otherwise only h is conditional)"
    )
    parser.add_argument("--outercouplingmlp", action="store_true", help="Use MLP instead of ResNet for coupling layers")
    parser.add_argument("--outercouplinglayers", type=int, default=2, help="Number of layers for coupling layers")
    parser.add_argument("--outercouplinghidden", type=int, default=100, help="Number of hidden units for coupling layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Use dropout")
    parser.add_argument("--pieepsilon", type=float, default=0.01, help="PIE epsilon term")
    parser.add_argument("--encoderblocks", type=int, default=5, help="Number of blocks in MFMFE encoder")
    parser.add_argument("--encoderhidden", type=int, default=100, help="Number of hidden units in MFMFE encoder")
    parser.add_argument("--encodermlp", action="store_true", help="Use MLP instead of ResNet for MFMFE encoder")
    parser.add_argument("--splinerange", default=3.0, type=float, help="Spline boundaries")
    parser.add_argument("--splinebins", default=8, type=int, help="Number of spline bins")
    parser.add_argument("--levels", type=int, default=3, help="Number of levels in multi-scale architectures for image data (for outer transformation)")
    parser.add_argument("--actnorm", action="store_true", help="Use actnorm in convolutional architecture")
    parser.add_argument("--batchnorm", action="store_true", help="Use batchnorm in ResNets")

    # Training
    parser.add_argument("--alternate", action="store_true", help="Use alternating training algorithm (e.g. MFMF-MD instead of MFMF-S)")
    parser.add_argument("--sequential", action="store_true", help="Use sequential training algorithm")
    parser.add_argument("--load", type=str, default=None, help="Model name to load rather than training from scratch, run is affixed automatically")
    parser.add_argument("--samplesize", type=int, default=None, help="If not None, number of samples used for training")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--subsets", type=int, default=1, help="Number of subsets per epoch in an alternating training")
    parser.add_argument("--batchsize", type=int, default=100, help="Batch size for everything except OT training")
    parser.add_argument("--genbatchsize", type=int, default=1000, help="Batch size for OT training")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Initial learning rate")
    parser.add_argument("--msefactor", type=float, default=1000.0, help="Reco error multiplier in loss")
    parser.add_argument("--addnllfactor", type=float, default=0.1, help="Negative log likelihood multiplier in loss for MFMF-S training")
    parser.add_argument("--nllfactor", type=float, default=1.0, help="Negative log likelihood multiplier in loss (except for MFMF-S training)")
    parser.add_argument("--sinkhornfactor", type=float, default=10.0, help="Sinkhorn divergence multiplier in loss")
    parser.add_argument("--weightdecay", type=float, default=1.0e-4, help="Weight decay")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient norm clipping parameter")
    # parser.add_argument("--doughl1reg", type=float, default=0.0, help="L1 reg on epsilon when learning epsilons for PIE")
    parser.add_argument("--nopretraining", action="store_true", help="Skip pretraining in MFMF-S training")
    parser.add_argument("--noposttraining", action="store_true", help="Skip posttraining in MFMF-S training")
    parser.add_argument("--prepie", action="store_true", help="Pretrain with PIE rather than on reco error (MFMF-S only)")
    parser.add_argument("--prepostfraction", type=int, default=3, help="Fraction of epochs reserved for pretraining and posttraining (MFMF-S only)")
    parser.add_argument("--validationsplit", type=float, default=0.25, help="Fraction of train data used for early stopping")
    parser.add_argument("--scandal", type=float, default=None, help="Activates SCANDAL training and sets prefactor of score MSE in loss")

    # Other settings
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")
    parser.add_argument("--dir", type=str, default="/scratch/jb6504/scandal-mf", help="Base directory of repo")
    parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")

    return parser.parse_args()


def make_training_kwargs(args, dataset):
    kwargs = {
        "dataset": dataset,
        "batch_size": args.batchsize,
        "initial_lr": args.lr,
        "scheduler": optim.lr_scheduler.CosineAnnealingLR,
        "clip_gradient": args.clip,
        "validation_split": args.validationsplit,
    }
    if args.weightdecay is not None:
        kwargs["optimizer_kwargs"] = {"weight_decay": float(args.weightdecay)}
    scandal_loss = [losses.score_mse] if args.scandal is not None else []
    scandal_label = ["score MSE"] if args.scandal is not None else []
    scandal_weight = [args.scandal] if args.scandal is not None else []

    return kwargs, scandal_loss, scandal_label, scandal_weight


def train_manifold_flow(args, dataset, model, simulator):
    """ MFMF-S training """

    assert not args.specified

    trainer = (
        ForwardTrainer(model)
        if simulator.parameter_dim() is None
        else ConditionalForwardTrainer(model)
        if args.scandal is None
        else SCANDALForwardTrainer(model)
    )
    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    if args.nopretraining or args.epochs // args.prepostfraction < 1:
        logger.info("Skipping pretraining phase")
        learning_curves = None
    elif args.prepie:
        logger.info("Starting training MF, phase 1: pretraining on PIE likelihood")
        learning_curves = trainer.train(
            loss_functions=[losses.nll] + scandal_loss,
            loss_labels=["NLL"] + scandal_label,
            loss_weights=[args.nllfactor] + scandal_weight,
            epochs=args.epochs // args.prepostfraction,
            callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_A{}.pt")],
            forward_kwargs={"mode": "pie"},
            **common_kwargs,
        )
        learning_curves = np.vstack(learning_curves).T
    else:
        logger.info("Starting training MF, phase 1: pretraining on reconstruction error")
        learning_curves = trainer.train(
            loss_functions=[losses.mse],
            loss_labels=["MSE"],
            loss_weights=[args.msefactor],
            epochs=args.epochs // args.prepostfraction,
            callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_A{}.pt")],
            forward_kwargs={"mode": "projection"},
            **common_kwargs,
        )
        learning_curves = np.vstack(learning_curves).T

    logger.info("Starting training MF, phase 2: mixed training")
    learning_curves_ = trainer.train(
        loss_functions=[losses.mse, losses.nll] + scandal_loss,
        loss_labels=["MSE", "NLL"] + scandal_label,
        loss_weights=[args.msefactor, args.addnllfactor] + scandal_weight,
        epochs=args.epochs - (2 - int(args.nopretraining) - int(args.noposttraining)) * (args.epochs // args.prepostfraction),
        parameters=model.parameters(),
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_B{}.pt")],
        forward_kwargs={"mode": "mf"},
        **common_kwargs,
    )
    learning_curves_ = np.vstack(learning_curves_).T
    learning_curves = learning_curves_ if learning_curves is None else np.vstack((learning_curves, learning_curves_))

    if args.nopretraining or args.epochs // args.prepostfraction < 1:
        logger.info("Skipping inner flow phase")
    else:
        logger.info("Starting training MF, phase 3: training only inner flow on NLL")
        learning_curves_ = trainer.train(
            loss_functions=[losses.mse, losses.nll] + scandal_loss,
            loss_labels=["MSE", "NLL"] + scandal_label,
            loss_weights=[0.0, args.nllfactor] + scandal_weight,
            epochs=args.epochs // args.prepostfraction,
            parameters=model.inner_transform.parameters(),
            callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_C{}.pt")],
            forward_kwargs={"mode": "mf-fixed-manifold"},
            **common_kwargs,
        )
        learning_curves = np.vstack((learning_curves, np.vstack(learning_curves_).T))

    return learning_curves


def train_specified_manifold_flow(args, dataset, model, simulator):
    """ FOM training """

    trainer = (
        ForwardTrainer(model)
        if simulator.parameter_dim() is None
        else ConditionalForwardTrainer(model)
        if args.scandal is None
        else SCANDALForwardTrainer(model)
    )
    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    logger.info("Starting training MF with specified manifold on NLL")
    learning_curves = trainer.train(
        loss_functions=[losses.mse, losses.nll] + scandal_loss,
        loss_labels=["MSE", "NLL"] + scandal_label,
        loss_weights=[0.0, args.nllfactor] + scandal_weight,
        epochs=args.epochs,
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_{}.pt")],
        forward_kwargs={"mode": "mf"},
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T

    return learning_curves


def train_manifold_flow_alternating(args, dataset, model, simulator):
    """ MFMF-A training """

    assert not args.specified

    trainer1 = ForwardTrainer(model) if simulator.parameter_dim() is None else ConditionalForwardTrainer(model)
    trainer2 = (
        ForwardTrainer(model)
        if simulator.parameter_dim() is None
        else ConditionalForwardTrainer(model)
        if args.scandal is None
        else SCANDALForwardTrainer(model)
    )
    metatrainer = AlternatingTrainer(model, trainer1, trainer2)

    meta_kwargs = {"dataset": dataset, "initial_lr": args.lr, "scheduler": optim.lr_scheduler.CosineAnnealingLR, "validation_split": args.validationsplit}
    if args.weightdecay is not None:
        meta_kwargs["optimizer_kwargs"] = {"weight_decay": float(args.weightdecay)}
    _, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    phase1_kwargs = {"forward_kwargs": {"mode": "projection"}, "clip_gradient": args.clip}
    phase2_kwargs = {"forward_kwargs": {"mode": "mf-fixed-manifold"}, "clip_gradient": args.clip}

    phase1_parameters = (
        list(model.outer_transform.parameters()) + list(model.encoder.parameters()) if args.algorithm == "emf" else model.outer_transform.parameters()
    )
    phase2_parameters = model.inner_transform.parameters()

    logger.info("Starting training MF, alternating between reconstruction error and log likelihood")
    learning_curves_ = metatrainer.train(
        loss_functions=[losses.mse, losses.nll] + scandal_loss,
        loss_function_trainers=[0, 1] + [1] if args.scandal is not None else [],
        loss_labels=["MSE", "NLL"] + scandal_label,
        loss_weights=[args.msefactor, args.nllfactor] + scandal_weight,
        epochs=args.epochs // 2,
        subsets=args.subsets,
        batch_sizes=[args.batchsize, args.batchsize],
        parameters=[phase1_parameters, phase2_parameters],
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_{}.pt")],
        trainer_kwargs=[phase1_kwargs, phase2_kwargs],
        **meta_kwargs,
    )
    learning_curves = np.vstack(learning_curves_).T

    return learning_curves


def train_manifold_flow_sequential(args, dataset, model, simulator):
    """ MFMF-A training """

    assert not args.specified

    trainer1 = ForwardTrainer(model) if simulator.parameter_dim() is None else ConditionalForwardTrainer(model)
    trainer2 = (
        ForwardTrainer(model)
        if simulator.parameter_dim() is None
        else ConditionalForwardTrainer(model)
        if args.scandal is None
        else SCANDALForwardTrainer(model)
    )
    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    logger.info("Starting training MF, phase 1: manifold training")
    learning_curves = trainer1.train(
        loss_functions=[losses.mse],
        loss_labels=["MSE"],
        loss_weights=[args.msefactor],
        epochs=args.epochs // 2,
        parameters=list(model.outer_transform.parameters()) + list(model.encoder.parameters())
        if args.algorithm == "emf"
        else model.outer_transform.parameters(),
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_A{}.pt")],
        forward_kwargs={"mode": "projection"},
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T

    logger.info("Starting training MF, phase 2: density training")
    learning_curves_ = trainer2.train(
        loss_functions=[losses.nll] + scandal_loss,
        loss_labels=["NLL"] + scandal_label,
        loss_weights=[args.nllfactor] + scandal_weight,
        epochs=args.epochs - (args.epochs // 2),
        parameters=model.inner_transform.parameters(),
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_B{}.pt")],
        forward_kwargs={"mode": "mf-fixed-manifold"},
        **common_kwargs,
    )
    learning_curves = np.vstack((learning_curves, np.vstack(learning_curves_).T))

    return learning_curves


def train_generative_adversarial_manifold_flow(args, dataset, model, simulator):
    """ MFMF-OT training """

    gen_trainer = AdversarialTrainer(model) if simulator.parameter_dim() is None else ConditionalAdversarialTrainer(model)
    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)
    common_kwargs["batch_size"] = args.genbatchsize

    logger.info("Starting training GAMF: Sinkhorn-GAN")

    callbacks_ = [callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_{}.pt")]
    if args.debug:
        callbacks_.append(callbacks.print_mf_weight_statistics())

    learning_curves_ = gen_trainer.train(
        loss_functions=[losses.make_sinkhorn_divergence()],
        loss_labels=["GED"],
        loss_weights=[args.sinkhornfactor],
        epochs=args.epochs,
        callbacks=callbacks_,
        compute_loss_variance=True,
        **common_kwargs,
    )

    learning_curves = np.vstack(learning_curves_).T
    return learning_curves


def train_generative_adversarial_manifold_flow_alternating(args, dataset, model, simulator):
    """ MFMF-OTA training """

    assert not args.specified

    gen_trainer = AdversarialTrainer(model) if simulator.parameter_dim() is None else ConditionalAdversarialTrainer(model)
    likelihood_trainer = (
        ForwardTrainer(model)
        if simulator.parameter_dim() is None
        else ConditionalForwardTrainer(model)
        if args.scandal is None
        else SCANDALForwardTrainer(model)
    )
    metatrainer = AlternatingTrainer(model, gen_trainer, likelihood_trainer)

    meta_kwargs = {"dataset": dataset, "initial_lr": args.lr, "scheduler": optim.lr_scheduler.CosineAnnealingLR, "validation_split": args.validationsplit}
    if args.weightdecay is not None:
        meta_kwargs["optimizer_kwargs"] = {"weight_decay": float(args.weightdecay)}
    _, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    phase1_kwargs = {"clip_gradient": args.clip}
    phase2_kwargs = {"forward_kwargs": {"mode": "mf-fixed-manifold"}, "clip_gradient": args.clip}

    phase1_parameters = model.parameters()
    phase2_parameters = model.inner_transform.parameters()

    logger.info("Starting training GAMF, alternating between Sinkhorn divergence and log likelihood")
    learning_curves_ = metatrainer.train(
        loss_functions=[losses.make_sinkhorn_divergence(), losses.nll] + scandal_loss,
        loss_function_trainers=[0, 1] + [1] if args.scandal is not None else [],
        loss_labels=["GED", "NLL"] + scandal_label,
        loss_weights=[args.sinkhornfactor, args.nllfactor] + scandal_weight,
        batch_sizes=[args.genbatchsize, args.batchsize],
        epochs=args.epochs // 2,
        parameters=[phase1_parameters, phase2_parameters],
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_{}.pt")],
        trainer_kwargs=[phase1_kwargs, phase2_kwargs],
        subsets=args.subsets,
        subset_callbacks=[callbacks.print_mf_weight_statistics()] if args.debug else None,
        **meta_kwargs,
    )
    learning_curves = np.vstack(learning_curves_).T

    return learning_curves


def train_slice_of_pie(args, dataset, model, simulator):
    """ SLICE training """

    trainer = (
        ForwardTrainer(model)
        if simulator.parameter_dim() is None
        else ConditionalForwardTrainer(model)
        if args.scandal is None
        else SCANDALForwardTrainer(model)
    )
    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    if args.nopretraining or args.epochs // 3 < 1:
        logger.info("Skipping pretraining phase")
        learning_curves = np.zeros((0, 2))
    else:
        logger.info("Starting training slice of PIE, phase 1: pretraining on reconstruction error")
        learning_curves = trainer.train(
            loss_functions=[losses.mse],
            loss_labels=["MSE"],
            loss_weights=[args.initialmsefactor],
            epochs=args.epochs // 3,
            callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_A{}.pt")],
            forward_kwargs={"mode": "projection"},
            **common_kwargs,
        )
        learning_curves = np.vstack(learning_curves).T

    logger.info("Starting training slice of PIE, phase 2: mixed training")
    learning_curves_ = trainer.train(
        loss_functions=[losses.mse, losses.nll] + scandal_loss,
        loss_labels=["MSE", "NLL"] + scandal_label,
        loss_weights=[args.initialmsefactor, args.initialnllfactor] + scandal_weight,
        epochs=args.epochs - (1 if args.nopretraining else 2) * (args.epochs // 3),
        parameters=model.inner_transform.parameters(),
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_B{}.pt")],
        forward_kwargs={"mode": "slice"},
        **common_kwargs,
    )
    learning_curves = np.vstack((learning_curves, np.vstack(learning_curves_).T))

    logger.info("Starting training slice of PIE, phase 3: training only inner flow on NLL")
    learning_curves_ = trainer.train(
        loss_functions=[losses.mse, losses.nll] + scandal_loss,
        loss_labels=["MSE", "NLL"] + scandal_loss,
        loss_weights=[args.msefactor, args.nllfactor] + scandal_weight,
        epochs=args.epochs // 3,
        parameters=model.inner_transform.parameters(),
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_C{}.pt")],
        forward_kwargs={"mode": "slice"},
        **common_kwargs,
    )
    learning_curves = np.vstack((learning_curves, np.vstack(learning_curves_).T))

    return learning_curves


def train_flow(args, dataset, model, simulator):
    """ AF training """
    trainer = (
        ForwardTrainer(model)
        if simulator.parameter_dim() is None
        else ConditionalForwardTrainer(model)
        if args.scandal is None
        else SCANDALForwardTrainer(model)
    )
    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)
    callbacks_ = [callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_{}.pt")]
    if simulator.is_image():
        callbacks_.append(callbacks.plot_sample_images(create_filename("training_plot", None, args)))

    logger.info("Starting training standard flow on NLL")
    learning_curves = trainer.train(
        loss_functions=[losses.nll] + scandal_loss,
        loss_labels=["NLL"] + scandal_label,
        loss_weights=[args.nllfactor] + scandal_weight,
        epochs=args.epochs,
        callbacks=callbacks_,
        **common_kwargs,
    )

    learning_curves = np.vstack(learning_curves).T
    return learning_curves


def train_pie(args, dataset, model, simulator):
    """ PIE training """
    trainer = (
        ForwardTrainer(model)
        if simulator.parameter_dim() is None
        else ConditionalForwardTrainer(model)
        if args.scandal is None
        else SCANDALForwardTrainer(model)
    )
    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    logger.info("Starting training PIE on NLL")
    learning_curves = trainer.train(
        loss_functions=[losses.nll] + scandal_loss,
        loss_labels=["NLL"] + scandal_label,
        loss_weights=[args.nllfactor] + scandal_weight,
        epochs=args.epochs,
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_{}.pt")],
        forward_kwargs={"mode": "pie"},
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T
    return learning_curves


# def train_dough(args, dataset, model, simulator):
#     """ PIE with variable epsilons training """
#     trainer = VarDimForwardTrainer(model) if simulator.parameter_dim() is None else ConditionalVarDimForwardTrainer(model)
#     common_kwargs, _, _, _ = make_training_kwargs(args, dataset)
#
#     logger.info("Starting training dough, phase 1: NLL without latent regularization")
#     learning_curves = trainer.train(
#         loss_functions=[losses.nll],
#         loss_labels=["NLL"],
#         loss_weights=[args.nllfactor],
#         epochs=args.epochs,
#         callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_{}.pt")],
#         l1=args.doughl1reg,
#         **common_kwargs,
#     )
#     learning_curves = np.vstack(learning_curves).T
#     return learning_curves


def train_model(args, dataset, model, simulator):
    """ Starts appropriate training """

    if args.algorithm == "pie":
        learning_curves = train_pie(args, dataset, model, simulator)
    elif args.algorithm == "flow":
        learning_curves = train_flow(args, dataset, model, simulator)
    elif args.algorithm == "slice":
        learning_curves = train_slice_of_pie(args, dataset, model, simulator)
    elif args.algorithm in ["mf", "emf"]:
        if args.alternate:
            learning_curves = train_manifold_flow_alternating(args, dataset, model, simulator)
        elif args.sequential:
            learning_curves = train_manifold_flow_sequential(args, dataset, model, simulator)
        elif args.specified:
            learning_curves = train_specified_manifold_flow(args, dataset, model, simulator)
        else:
            learning_curves = train_manifold_flow(args, dataset, model, simulator)
    elif args.algorithm == "gamf":
        if args.alternate:
            learning_curves = train_generative_adversarial_manifold_flow_alternating(args, dataset, model, simulator)
        else:
            learning_curves = train_generative_adversarial_manifold_flow(args, dataset, model, simulator)
    # elif args.algorithm == "dough":
    #     learning_curves = train_dough(args, dataset, model, simulator)
    else:
        raise ValueError("Unknown algorithm %s", args.algorithm)

    return learning_curves


if __name__ == "__main__":
    # Logger
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO
    )
    logger.info("Hi!")
    logger.debug("Starting train.py with arguments %s", args)

    create_modelname(args)

    if args.load is None:
        logger.info("Training model %s with algorithm %s on data set %s", args.modelname, args.algorithm, args.dataset)
    else:
        logger.info("Loading model %s and training it as %s with algorithm %s on data set %s", args.load, args.modelname, args.algorithm, args.dataset)

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Data
    simulator = load_simulator(args)
    dataset = load_training_dataset(simulator, args)

    # Model
    model = create_model(args, simulator)

    # Maybe load pretrained model
    if args.load is not None:
        args_ = copy.deepcopy(args)
        args_.modelname = args.load
        if args_.i > 0:
            args_.modelname += "_run{}".format(args_.i)
        logger.info("Loading model %s", args_.modelname)
        model.load_state_dict(torch.load(create_filename("model", None, args_), map_location=torch.device("cpu")))

    # Train and save
    learning_curves = train_model(args, dataset, model, simulator)

    # Save
    logger.info("Saving model")
    torch.save(model.state_dict(), create_filename("model", None, args))
    np.save(create_filename("learning_curve", None, args), learning_curves)

    logger.info("All done! Have a nice day!")
