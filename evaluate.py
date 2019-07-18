#! /usr/bin/env python

import numpy as np
import logging
import sys
import torch
import argparse
from torch.utils.data import DataLoader
from torch.nn import MSELoss

sys.path.append("../")

from aef.models.autoencoding_flow import TwoStepAutoencodingFlow
from aef.trainer import NumpyDataset
from aef import losses
from generate_gaussian_data import true_logp

logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.DEBUG,
)


def eval_on_test_data(
    model_filename,
    dataset,
    data_dim,
    flow_latent_dim,
    flow_steps_inner=5,
    flow_steps_outer=5,
    batch_size=1024,
    base_dir=".",
):
    logging.info("Evaluating %s on test data", model_filename)

    # Prepare evaluation
    run_on_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.float
    if run_on_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Dataset
    if dataset == "tth":
        x = np.load("{}/data/tth/x_test.npy".format(base_dir))
        x_means = np.mean(x, axis=0)
        x_stds = np.std(x, axis=0)
        x = (x - x_means[np.newaxis, :]) / x_stds[np.newaxis, :]
        y = np.zeros(x.shape[0])
        data = NumpyDataset(x, y)
        data_dim = 48
        n_samples = x.shape[0]
    elif dataset == "gaussian":
        assert data_dim is not None
        transform = np.load("{}/data/gaussian/gaussian_transform.npy".format(base_dir))
        true_latent_dim = 8
        x = np.load("{}/data/gaussian/gaussian_{}_{}_x_train.npy".format(base_dir, true_latent_dim, data_dim))
        y = true_logp(x, 0.001, 8, data_dim, transform, True)
        data = NumpyDataset(x, y)
    else:
        raise NotImplementedError("Unknown dataset {}".format(dataset))

    # Dataloader
    dataloader = DataLoader(data, batch_size=batch_size, pin_memory=run_on_gpu)

    # Model
    ae = TwoStepAutoencodingFlow(
        data_dim=data_dim,
        latent_dim=flow_latent_dim,
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
    mse_reco = 0.0
    mse_log_likelihood = 0.0

    # Evaluation loop
    with torch.no_grad():
        for x, log_prob_true in dataloader:
            x = x.view(x.size(0), -1)
            x = x.to(device, dtype)
            log_prob_true = log_prob_true.to(device, dtype)
            x_reco, log_prob, _ = ae(x)

            mse_reco += losses.mse(x_reco, x, log_prob).item()
            nll += losses.nll(x_reco, x, log_prob).item()

            if dataset == "gaussian":
                mse_log_likelihood += MSELoss()(log_prob, log_prob_true).item()

    # Copy back tensors to CPU
    if run_on_gpu:
        try:
            nll = nll.cpu()
            mse_reco = mse_reco.cpu()
            mse_log_likelihood = mse_log_likelihood.cpu()
        except AttributeError:
            pass
    nll = nll / len(dataloader)
    mse_reco = mse_reco / len(dataloader)
    mse_log_likelihood = mse_log_likelihood / len(dataloader)

    logging.info("Result: - log likelihood = %s, reco MSE = %s, log likelihood MSE = %s", nll, mse_reco, mse_log_likelihood)

    return nll, mse_reco, mse_log_likelihood


def eval_generated_data(
    model_filename,
    dataset,
    data_dim,
    flow_latent_dim,
    flow_steps_inner=5,
    flow_steps_outer=5,
    n = 10000,
    base_dir=".",
):
    logging.info("Generating and evaluating data from %s", model_filename)

    # Prepare evaluation
    run_on_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.float
    if run_on_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Dataset
    if dataset == "tth":
        raise RuntimeError("Cannot evaluate true likelihood for ttH dataset")
    elif dataset == "gaussian":
        assert data_dim is not None
        true_latent_dim = 8
    else:
        raise NotImplementedError("Unknown dataset {}".format(dataset))

    # Model
    ae = TwoStepAutoencodingFlow(
        data_dim=data_dim,
        latent_dim=flow_latent_dim,
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

    # Generate data
    x = ae.sample(n=n)
    if run_on_gpu:
        try:
            x = x.cpu()
        except AttributeError:
            pass
    x = x.detach().numpy()

    # Calculate true likelihood of generated data
    transform = np.load("{}/data/gaussian/gaussian_transform.npy".format(base_dir))
    nll = - true_logp(x=x, epsilon=0.001, latent_dim=true_latent_dim, data_dim=data_dim, transform=transform)
    nll = np.mean(nll, axis=0)

    logging.info("Result: - log likelihood = %s", nll)

    return nll



def eval_loop_gaussian(
    result_filename="gaussian",
    model_filename="gaussian_{}_{}_{}",
    flow_latent_dims=(2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,64,128),
    data_dims=(8,16,32,64,128),
    flow_steps_inner=5,
    flow_steps_outer=5,
    batch_size=1024,
    n_gen=10000,
    base_dir=".",
):
    true_latent_dim = 8

    # Output
    flow_latent_dims_out = []
    data_dims_out = []
    nll_tests, mse_reco_tests, mse_log_likelihood_tests, nll_gens = [], [], [], []

    for data_dim in data_dims:
        for flow_latent_dim in flow_latent_dims:
            if flow_latent_dim > data_dim:
                break

            filename = model_filename.format(true_latent_dim, data_dim, flow_latent_dim)

            nll_gen = eval_generated_data(
                filename,
                dataset="gaussian",
                data_dim=data_dim,
                flow_latent_dim=flow_latent_dim,
                flow_steps_inner=flow_steps_inner,
                flow_steps_outer=flow_steps_outer,
                base_dir=base_dir,
                n=n_gen
            )

            nll, mse_reco, mse_logp = eval_on_test_data(
                filename,
                dataset="gaussian",
                data_dim=data_dim,
                flow_latent_dim=flow_latent_dim,
                flow_steps_inner=flow_steps_inner,
                flow_steps_outer=flow_steps_outer,
                batch_size=batch_size,
                base_dir=base_dir,
            )

            data_dims_out.append(data_dim)
            flow_latent_dims_out.append(flow_latent_dim)
            nll_tests.append(nll)
            mse_reco_tests.append(mse_reco)
            nll_gens.append(nll_gen)
            mse_log_likelihood_tests.append(mse_logp)

    data_dims_out = np.array(data_dims_out, dtype=np.int)
    flow_latent_dims_out = np.array(flow_latent_dims_out, dtype=np.int)
    nll_tests = np.array(nll_tests)
    mse_reco_tests = np.array(mse_reco_tests)
    nll_gens = np.array(nll_gens)
    mse_log_likelihood_tests = np.array(mse_log_likelihood_tests)

    np.save("{}/data/results/data_dims_{}.npy".format(base_dir, result_filename), data_dims_out)
    np.save("{}/data/results/latent_dims_{}.npy".format(base_dir, result_filename), flow_latent_dims_out)
    np.save("{}/data/results/nll_test_{}.npy".format(base_dir, result_filename), nll_tests)
    np.save("{}/data/results/mse_reco_test_{}.npy".format(base_dir, result_filename), mse_reco_tests)
    np.save("{}/data/results/mse_log_likelihood_test_{}.npy".format(base_dir, result_filename), mse_log_likelihood_tests)
    np.save("{}/data/results/nll_gen_{}.npy".format(base_dir, result_filename), nll_gens)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=str)
    parser.add_argument("--dir", type=str, default="/Users/johannbrehmer/work/projects/ae_flow/autoencoded-flow")
    parser.add_argument("--data", type=int, default=(8,16,32,64,128))
    return parser.parse_args()


if __name__ == "__main__":
    logging.info("Hi!")
    args = parse_args()
    if isinstance(args.data, int):
        args.data = (args.data,)
    eval_loop_gaussian(base_dir=args.dir, data_dims=args.data, result_filename=args.result)
    logging.info("All done! Have a nice day!")
