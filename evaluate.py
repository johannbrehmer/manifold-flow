#! /usr/bin/env python

import numpy as np
import logging
import sys
import torch
import argparse
from torch.utils.data import DataLoader

sys.path.append("../")

from aef.models.autoencoding_flow import TwoStepAutoencodingFlow
from aef.trainer import AutoencodingFlowTrainer, NumpyDataset
from aef import losses

logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.DEBUG,
)


def evaluation_loop(
    result_filename,
    model_filename,
    latent_dims,
    dataset="tth",
    flow_steps_inner=5,
    flow_steps_outer=5,
    batch_size=1024,
    base_dir=".",
):
    nlls, mses = [], []
    for latent_dim in latent_dims:
        filename = model_filename.format(latent_dim)
        nll, mse = eval_model(
            filename,
            dataset=dataset,
            latent_dim=latent_dim,
            flow_steps_inner=flow_steps_inner,
            flow_steps_outer=flow_steps_outer,
            batch_size=batch_size,
            base_dir=base_dir,
        )
        nlls.append(nll)
        mses.append(mse)

    nlls = np.array(nlls)
    mses = np.array(mses)
    latent_dims = np.array(latent_dims)

    np.save("{}/data/results/nll_{}.npy".format(base_dir, result_filename), nlls)
    np.save("{}/data/results/mse_{}.npy".format(base_dir, result_filename), mses)
    np.save(
        "{}/data/results/latent_dims_{}.npy".format(base_dir, result_filename), latent_dims
    )


def eval_model(
    model_filename,
    dataset="tth",
    latent_dim=10,
    flow_steps_inner=5,
    flow_steps_outer=5,
    batch_size=1024,
    base_dir=".",
):
    # Prepare evaluation
    run_on_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.float

    # Dataset
    if dataset == "tth":
        x = np.load("{}/data/tth/x_test.npy".format(base_dir))
        x_means = np.mean(x, axis=0)
        x_stds = np.std(x, axis=0)
        x = (x - x_means[np.newaxis, :]) / x_stds[np.newaxis, :]
        y = np.ones(x.shape[0])
        data = NumpyDataset(x, y)
        data_dim = 48
        n_samples = x.shape[0]
    else:
        raise NotImplementedError("Unknown dataset {}".format(dataset))

    # Dataloader
    dataloader = DataLoader(data, batch_size=batch_size, pin_memory=run_on_gpu)

    # Model
    ae = TwoStepAutoencodingFlow(
        data_dim=data_dim,
        latent_dim=latent_dim,
        steps_inner=flow_steps_inner,
        steps_outer=flow_steps_outer,
    )

    # Load state dict
    ae.load_state_dict(
        torch.load(
            "{}/data/models/{}.pt".format(base_dir, model_filename), map_location="cpu"
        )
    )
    ae = ae.to(device, dtype)
    ae.eval()

    # Metrics
    nll = 0.0
    mse = 0.0

    # Evaluation loop
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.view(x.size(0), -1)
            x = x.to(device, dtype)
            x_reco, log_prob, _ = ae(x)

            mse += losses.mse(x_reco, x, log_prob).item()
            nll += losses.nll(x_reco, x, log_prob).item()

    # Copy back tensors to CPU
    if run_on_gpu:
        nll = nll.cpu()
        mse = mse.cpu()
    nll = nll / len(dataloader)
    mse = mse / len(dataloader)

    return nll, mse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strong lensing experiments: simulation"
    )
    parser.add_argument("results", type=str, help="Result name.")
    parser.add_argument("model", type=str, help="Model name.")
    parser.add_argument("--dataset", type=str, default="tth", choices=["tth"])
    parser.add_argument("--latentmin", type=int, default=2)
    parser.add_argument("--latentmax", type=int, default=32)
    parser.add_argument("--latentsteps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dir", type=str, default=".")
    return parser.parse_args()


if __name__ == "__main__":
    logging.info("Hi!")

    args = parse_args()

    evaluation_loop(
        args.results,
        model_filename=args.model,
        latent_dims=list(range(args.latentmin, args.latentmax + 1, args.latentsteps)),
        dataset=args.dataset,
        flow_steps_inner=args.steps,
        flow_steps_outer=args.steps,
        base_dir=args.dir,
    )

    logging.info("All done! Have a nice day!")
