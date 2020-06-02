#! /usr/bin/env python

""" Top-level script for simple timing experiments """

import numpy as np
import logging
import sys
import torch
import argparse
import time

sys.path.append("../")

from utils import create_filename
from architectures import create_model

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", type=str, default="mf", choices=["flow", "pie", "mf"])

    parser.add_argument("--conditionalouter", action="store_true")
    parser.add_argument("--modellatentdim", type=int, default=8)
    parser.add_argument("--outertransform", type=str, default="affine-coupling")
    parser.add_argument("--innertransform", type=str, default="affine-coupling")
    parser.add_argument("--outerlayers", type=int, default=4)
    parser.add_argument("--innerlayers", type=int, default=8)
    parser.add_argument("--outercouplingmlp", action="store_true")
    parser.add_argument("--outercouplinglayers", type=int, default=3)
    parser.add_argument("--outercouplinghidden", type=int, default=256)

    parser.add_argument("--datadims", nargs="+", type=int, default=[10, 20, 50, 100, 200])
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)

    parser.add_argument("--dir", type=str, default="/scratch/jb6504/manifold-flow")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def timing(args):
    logger.info(
        "Timing algorithm %s with %s outer layers with transformation %s and %s inner layers with transformation %s",
        args.algorithm,
        args.outerlayers,
        args.outertransform,
        args.innerlayers,
        args.innertransform,
    )

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.DoubleTensor")

    # Loop over data dims
    all_times = []
    for datadim in args.datadims:
        logger.info("Starting timing for %s-dimensional data", datadim)
        args.datadim = datadim

        # Data
        data = torch.randn(args.batchsize, datadim)
        data.requires_grad = True

        # Model
        model = create_model(args, context_features=None)
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda"))

        # Time forward pass
        times = []
        for _ in range(args.repeats):
            time_before = time.time()
            _ = model(data)
            times.append(time.time() - time_before)

        logger.info("Mean time: %s s", np.mean(times))

        all_times.append(times)

    # Save results
    logger.info("Saving results")
    np.save(create_filename("timing", None, args), all_times)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO)
    logger.info("Hi!")
    timing(args)
    logger.info("All done! Have a nice day!")
