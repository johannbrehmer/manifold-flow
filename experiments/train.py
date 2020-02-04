#! /usr/bin/env python

import numpy as np
import logging
import sys
import torch
import argparse
import copy
from torch import optim

sys.path.append("../")

from manifold_flow.training import ManifoldFlowTrainer, losses, ConditionalManifoldFlowTrainer, callbacks, GenerativeTrainer, ConditionalGenerativeTrainer
from manifold_flow.training import VariableDimensionManifoldFlowTrainer, ConditionalVariableDimensionManifoldFlowTrainer
from experiments.utils.loading import load_training_dataset, load_simulator
from experiments.utils.names import create_filename, create_modelname, ALGORITHMS, SIMULATORS
from experiments.utils.models import create_model

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # What what what
    parser.add_argument("--modelname", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default="flow", choices=ALGORITHMS)
    parser.add_argument("--dataset", type=str, default="spherical_gaussian", choices=SIMULATORS)

    # Dataset details
    parser.add_argument("--truelatentdim", type=int, default=2)
    parser.add_argument("--datadim", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.01)

    # Model details
    parser.add_argument("--modellatentdim", type=int, default=2)
    parser.add_argument("--specified", action="store_true")
    parser.add_argument("--outertransform", type=str, default="rq-coupling")
    parser.add_argument("--innertransform", type=str, default="rq-coupling")
    parser.add_argument("--lineartransform", type=str, default="permutation")
    parser.add_argument("--outerlayers", type=int, default=5)
    parser.add_argument("--innerlayers", type=int, default=5)
    parser.add_argument("--conditionalouter", action="store_true")
    parser.add_argument("--outercouplingmlp", action="store_true")
    parser.add_argument("--outercouplinglayers", type=int, default=2)
    parser.add_argument("--outercouplinghidden", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pieepsilon", type=float, default=0.01)
    parser.add_argument("--ged", action="store_true")
    parser.add_argument("--encoderblocks", type=int, default=5)
    parser.add_argument("--encoderhidden", type=int, default=100)
    parser.add_argument("--encodermlp", action="store_true")

    # Training
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--genbatchsize", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--msefactor", type=float, default=1000.0)
    parser.add_argument("--addnllfactor", type=float, default=0.1)
    parser.add_argument("--nllfactor", type=float, default=1.0)
    parser.add_argument("--sinkhornfactor", type=float, default=10.0)
    parser.add_argument("--samplesize", type=int, default=None)
    parser.add_argument("--l2reg", type=float, default=None)
    parser.add_argument("--doughl1reg", type=float, default=0.0)
    parser.add_argument("--nopretraining", action="store_true")
    parser.add_argument("--noposttraining", action="store_true")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--prepie", action="store_true")
    parser.add_argument("--prepostfraction", type=int, default=3)

    # Other settings
    parser.add_argument("--dir", type=str, default="../")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def train_manifold_flow(args, dataset, model, simulator):
    trainer = ManifoldFlowTrainer(model) if simulator.parameter_dim() is None else ConditionalManifoldFlowTrainer(model)
    common_kwargs = {
        "dataset": dataset,
        "batch_size": args.batchsize,
        "initial_lr": args.lr,
        "scheduler": optim.lr_scheduler.CosineAnnealingLR,
        "clip_gradient": args.clip,
    }
    if args.l2reg is not None:
        common_kwargs["optimizer_kwargs"] = {"weight_decay": float(args.l2reg)}

    if args.specified:
        logger.info("Starting training MF with specified manifold on NLL")
        learning_curves = trainer.train(
            loss_functions=[losses.mse, losses.nll],
            loss_labels=["MSE", "NLL"],
            loss_weights=[0.0, args.nllfactor],
            epochs=args.epochs,
            callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_{}.pt")],
            forward_kwargs={"mode": "mf"},
            **common_kwargs,
        )
        learning_curves = np.vstack(learning_curves).T
    else:
        if args.nopretraining or args.epochs // args.prepostfraction < 1:
            logger.info("Skipping pretraining phase")
            learning_curves = None
        elif args.prepie:
            logger.info("Starting training MF, phase 1: pretraining on PIE likelihood")
            learning_curves = trainer.train(
                loss_functions=[losses.nll],
                loss_labels=["NLL"],
                loss_weights=[args.nllfactor],
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
            loss_functions=[losses.mse, losses.nll],
            loss_labels=["MSE", "NLL"],
            loss_weights=[args.msefactor, args.addnllfactor],
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
                loss_functions=[losses.mse, losses.nll],
                loss_labels=["MSE", "NLL"],
                loss_weights=[0.0, args.nllfactor],
                epochs=args.epochs // args.prepostfraction,
                parameters=model.inner_transform.parameters(),
                callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_C{}.pt")],
                forward_kwargs={"mode": "pie" if args.algorithm == "mf" else "mf"},
                **common_kwargs,
            )
            learning_curves_ = np.vstack(learning_curves_).T
            learning_curves = np.vstack((learning_curves, learning_curves_))

    return learning_curves


def train_generative_adversarial_manifold_flow(args, dataset, model, simulator):
    # trainer = ManifoldFlowTrainer(model) if simulator.parameter_dim() is None else ConditionalManifoldFlowTrainer(model)
    gen_trainer = GenerativeTrainer(model) if simulator.parameter_dim() is None else ConditionalGenerativeTrainer(model)
    common_kwargs = {"dataset": dataset, "initial_lr": args.lr, "scheduler": optim.lr_scheduler.CosineAnnealingLR, "clip_gradient": args.clip}
    if args.l2reg is not None:
        common_kwargs["optimizer_kwargs"] = {"weight_decay": float(args.l2reg)}

    logger.info("Starting training GAMF: Sinkhorn-GAN")
    learning_curves_ = gen_trainer.train(
        loss_functions=[losses.make_sinkhorn_divergence()],
        loss_labels=["GED"],
        loss_weights=[args.sinkhornfactor],
        epochs=args.epochs,
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_{}.pt")],
        batch_size=args.genbatchsize,
        compute_loss_variance=True,
        **common_kwargs,
    )

    learning_curves = np.vstack(learning_curves_).T
    return learning_curves


def train_hybrid(args, dataset, model, simulator):
    trainer = ManifoldFlowTrainer(model) if simulator.parameter_dim() is None else ConditionalManifoldFlowTrainer(model)
    gen_trainer = GenerativeTrainer(model) if simulator.parameter_dim() is None else ConditionalGenerativeTrainer(model)
    common_kwargs = {"dataset": dataset, "initial_lr": args.lr, "scheduler": optim.lr_scheduler.CosineAnnealingLR, "clip_gradient": args.clip}
    if args.l2reg is not None:
        common_kwargs["optimizer_kwargs"] = {"weight_decay": float(args.l2reg)}

    logger.info("Starting training hybrid, phase 1: pretraining on reconstruction error")
    learning_curves = trainer.train(
        loss_functions=[losses.mse],
        loss_labels=["MSE"],
        loss_weights=[args.msefactor],
        epochs=args.epochs // 4,
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_A{}.pt")],
        forward_kwargs={"mode": "projection"},
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T

    if args.ged and simulator.parameter_dim() is None:
        logger.info("Starting training hybrid: OT-GAN")
        learning_curves_ = gen_trainer.train(
            loss_functions=[losses.make_generalized_energy_distance()],
            loss_labels=["GED"],
            loss_weights=[args.sinkhornfactor],
            epochs=args.epochs - 3 * args.epochs // 4,
            callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_B{}.pt")],
            batch_size=args.genbatchsize,
            compute_loss_variance=True,
            **common_kwargs,
        )
    elif args.ged:
        logger.info("Starting training hybrid: Conditional OT-GAN")
        learning_curves_ = gen_trainer.train(
            loss_functions=[losses.make_conditional_generalized_energy_distance()],
            loss_labels=["GED"],
            loss_weights=[args.sinkhornfactor],
            epochs=args.epochs - 3 * args.epochs // 4,
            callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_B{}.pt")],
            batch_size=args.genbatchsize,
            compute_loss_variance=True,
            **common_kwargs,
        )
    else:
        logger.info("Starting training hybrid: Sinkhorn-GAN")
        learning_curves_ = gen_trainer.train(
            loss_functions=[losses.make_sinkhorn_divergence()],
            loss_labels=["GED"],
            loss_weights=[args.sinkhornfactor],
            epochs=args.epochs - 3 * args.epochs // 4,
            callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_B{}.pt")],
            batch_size=args.genbatchsize,
            compute_loss_variance=True,
            **common_kwargs,
        )
    learning_curves = np.vstack((learning_curves, learning_curves_))

    logger.info("Starting training hybrid, phase 3: mixed training")
    learning_curves_ = trainer.train(
        loss_functions=[losses.mse, losses.nll],
        loss_labels=["MSE", "NLL"],
        loss_weights=[args.msefactor, args.addnllfactor],
        epochs=args.epochs - 3 * args.epochs // 4,
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_C{}.pt")],
        forward_kwargs={"mode": "mf"},
        **common_kwargs,
    )
    learning_curves_ = np.vstack(learning_curves_).T
    learning_curves = np.vstack((learning_curves, learning_curves_))

    logger.info("Starting training MF, phase 4: training only inner flow on NLL")
    learning_curves_ = trainer.train(
        loss_functions=[losses.mse, losses.nll],
        loss_labels=["MSE", "NLL"],
        loss_weights=[args.msefactor, args.nllfactor],
        epochs=args.epochs // 4,
        parameters=model.inner_transform.parameters(),
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_D{}.pt")],
        forward_kwargs={"mode": "pie"},
        **common_kwargs,
    )
    learning_curves_ = np.vstack(learning_curves_).T
    learning_curves = np.vstack((learning_curves, learning_curves_))

    return learning_curves


def train_slice_of_pie(args, dataset, model, simulator):
    trainer = ManifoldFlowTrainer(model) if simulator.parameter_dim() is None else ConditionalManifoldFlowTrainer(model)
    common_kwargs = {
        "dataset": dataset,
        "batch_size": args.batchsize,
        "initial_lr": args.lr,
        "scheduler": optim.lr_scheduler.CosineAnnealingLR,
        "clip_gradient": args.clip,
    }
    if args.l2reg is not None:
        common_kwargs["optimizer_kwargs"] = {"weight_decay": float(args.l2reg)}

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
        loss_functions=[losses.mse, losses.nll],
        loss_labels=["MSE", "NLL"],
        loss_weights=[args.initialmsefactor, args.initialnllfactor],
        epochs=args.epochs - (1 if args.nopretraining else 2) * (args.epochs // 3),
        parameters=model.inner_transform.parameters(),
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_B{}.pt")],
        forward_kwargs={"mode": "slice"},
        **common_kwargs,
    )
    learning_curves_ = np.vstack(learning_curves_).T
    learning_curves = np.vstack((learning_curves, learning_curves_))

    logger.info("Starting training slice of PIE, phase 3: training only inner flow on NLL")
    learning_curves_ = trainer.train(
        loss_functions=[losses.mse, losses.nll],
        loss_labels=["MSE", "NLL"],
        loss_weights=[args.msefactor, args.nllfactor],
        epochs=args.epochs // 3,
        parameters=model.inner_transform.parameters(),
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_C{}.pt")],
        forward_kwargs={"mode": "slice"},
        **common_kwargs,
    )
    learning_curves_ = np.vstack(learning_curves_).T
    learning_curves = np.vstack((learning_curves, learning_curves_))

    return learning_curves


def train_flow(args, dataset, model, simulator):
    trainer = ManifoldFlowTrainer(model) if simulator.parameter_dim() is None else ConditionalManifoldFlowTrainer(model)
    logger.info("Starting training standard flow on NLL")
    common_kwargs = {
        "dataset": dataset,
        "batch_size": args.batchsize,
        "initial_lr": args.lr,
        "scheduler": optim.lr_scheduler.CosineAnnealingLR,
        "clip_gradient": args.clip,
    }
    if args.l2reg is not None:
        common_kwargs["optimizer_kwargs"] = {"weight_decay": float(args.l2reg)}

    learning_curves = trainer.train(
        loss_functions=[losses.nll],
        loss_labels=["NLL"],
        loss_weights=[args.nllfactor],
        epochs=args.epochs,
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_{}.pt")],
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T
    return learning_curves


def train_pie(args, dataset, model, simulator):
    trainer = ManifoldFlowTrainer(model) if simulator.parameter_dim() is None else ConditionalManifoldFlowTrainer(model)
    logger.info("Starting training PIE on NLL")
    common_kwargs = {
        "dataset": dataset,
        "batch_size": args.batchsize,
        "initial_lr": args.lr,
        "scheduler": optim.lr_scheduler.CosineAnnealingLR,
        "clip_gradient": args.clip,
    }
    if args.l2reg is not None:
        common_kwargs["optimizer_kwargs"] = {"weight_decay": float(args.l2reg)}

    learning_curves = trainer.train(
        loss_functions=[losses.nll],
        loss_labels=["NLL"],
        loss_weights=[args.nllfactor],
        epochs=args.epochs,
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_{}.pt")],
        forward_kwargs={"mode": "pie"},
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T
    return learning_curves


def train_dough(args, dataset, model, simulator):
    trainer = VariableDimensionManifoldFlowTrainer(model) if simulator.parameter_dim() is None else ConditionalVariableDimensionManifoldFlowTrainer(model)
    common_kwargs = {
        "dataset": dataset,
        "batch_size": args.batchsize,
        "initial_lr": args.lr,
        "scheduler": optim.lr_scheduler.CosineAnnealingLR,
        "clip_gradient": args.clip,
    }
    if args.l2reg is not None:
        common_kwargs["optimizer_kwargs"] = {"weight_decay": float(args.l2reg)}

    logger.info("Starting training dough, phase 1: NLL without latent regularization")
    learning_curves = trainer.train(
        loss_functions=[losses.nll],
        loss_labels=["NLL"],
        loss_weights=[args.nllfactor],
        epochs=args.epochs,
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_{}.pt")],
        l1=args.doughl1reg,
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T

    # logger.info("Starting training slice of PIE, phase 2: NLL with latent regularization")
    # learning_curves_ = trainer.train(
    #     loss_functions=[losses.nll],
    #     loss_labels=["NLL"],
    #     loss_weights=[args.nllfactor],
    #     epochs=args.epochs - args.epochs // 2,
    #     callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args)[:-3] + "_epoch_B{}.pt")],
    #     l1=args.doughl1reg,
    #     **common_kwargs,
    # )
    # learning_curves_ = np.vstack(learning_curves_).T
    # learning_curves = np.vstack((learning_curves, learning_curves_))
    return learning_curves


if __name__ == "__main__":
    # Logger
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO
    )
    logger.info("Hi!")

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

    logger.info("Parameters: %s", simulator.parameter_dim())

    # Model
    model = create_model(args, simulator)

    # Maybe load pretrained model
    if args.load is not None:
        logger.info("Loading model %s")
        args_ = copy.deepcopy(args)
        args_.modelname = args.load
        model.load_state_dict(torch.load(create_filename("model", None, args_), map_location=torch.device("cpu")))

    # Train
    if args.algorithm == "pie":
        learning_curves = train_pie(args, dataset, model, simulator)
    elif args.algorithm == "flow":
        learning_curves = train_flow(args, dataset, model, simulator)
    elif args.algorithm == "slice":
        learning_curves = train_slice_of_pie(args, dataset, model, simulator)
    elif args.algorithm in ["mf", "emf"]:
        learning_curves = train_manifold_flow(args, dataset, model, simulator)
    elif args.algorithm == "gamf":
        learning_curves = train_generative_adversarial_manifold_flow(args, dataset, model, simulator)
    elif args.algorithm == "hybrid":
        learning_curves = train_hybrid(args, dataset, model, simulator)
    elif args.algorithm == "dough":
        learning_curves = train_dough(args, dataset, model, simulator)
    else:
        raise ValueError("Unknown algorithm %s", args.algorithm)

    # Save
    logger.info("Saving model")
    torch.save(model.state_dict(), create_filename("model", None, args))
    np.save(create_filename("learning_curve", None, args), learning_curves)

    logger.info("All done! Have a nice day!")
