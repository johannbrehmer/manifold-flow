# Manifold-learning flows

*Johann Brehmer and Kyle Cranmer 2019-2020*

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Experiments with flows on manifolds.

## Introduction

## Using the code

### Getting started

Please make sure your Python environment satisfies the requirements in the [environment.yml](environment.yml). To use the MLF-OT algorithm, please also follow the [installation instructions for geomloss](https://www.kernel-operations.io/geomloss/api/install.html).

### Data sets

Data set | Data dimension | Manifold dimension | Model parameters | Arguments to `generate_data.py`, `train.py`, and `evaluate.py`
--- | --- | --- | --- | ---
Gaussian on an `n`-sphere | `n` | `d` | - |  `--dataset spherical_gaussian --truelatentdim n --datadim d --epsilon eps`
Conditional Gaussian on a `n`-sphere | `n` | `d` | 2 | `--dataset conditional_spherical_gaussian --truelatentdim n --datadim d`
Mixture model on a polynomial manifold | 3 | 2 | 1 | `--dataset power`
Particle physics data (48D) | 48 | 14 | 2 | `--dataset lhc`
Particle physics data (40D, no angular features) | 2 | 2 | 2 | `--dataset lhc2d`
Particle physics data (2D summary stats) | 2 | 2 | 2 | `--dataset lhc2d`
CIFAR10 | 3 * 32 * 32 | ? | - | `--dataset cifar10`
ImageNet | 3 * 64 * 64 | ? | - | `--dataset imagenet`


### Simulating data

Necessary for the first three data sets in the table above. See [experiments/generate_data.py -h](experiments/generate_data.py).


### Training 

See [experiments/train.py -h](experiments/train.py). Note that the algorithms have different internal names from the acronyms in the paper:

Algorithm | Acronym in paper | Arguments to `train.py`
--- | --- | ---
Ambient flow | AF | `--algorithm flow`
Flow on manifold | FOM | `--algorithm mf --specified`
Pseudo-invertible encoder | PIE | `--algorithm pie`
Manifold-modeling flow, simultaneous training | MFMF-S | `--algorithm mf`
Manifold-modeling flow, alternating training | MFMF-A | `--algorithm mf --alternate`
Manifold-modeling flow, Optimal Transport training | MFMF-OT | `--algorithm gamf`
Manifold-modeling flow, alternating Optimal Transport training | MFMF-OTA | `--algorithm gamf --alternate`
Manifold-modeling flow with sep. encoder, simultaneous training | MFMF'-S | `--algorithm emf`
Manifold-modeling flow with sep. encoder, alternating training schedule | MFMF'-A | `--algorithm emf --alternate`


### Evaluation 

See [experiments/evaluate.py -h](experiments/evaluate.py) and the notebooks in [experiments/notebooks](experiments/notebooks). Note that the algorithms have different internal names from the acronyms in the paper:

Algorithm | Acronym in paper | Arguments to `evaluate.py`
--- | --- | ---
Ambient flow | AF | `--algorithm flow`
Flow on manifold | FOM | `--algorithm mf --specified`
Pseudo-invertible encoder | PIE | `--algorithm pie`
Manifold-modeling flow (except OT training) | MFMF | `--algorithm mf`
Manifold-modeling flow, Optimal Transport training | MFMF-OT | `--algorithm gamf`
Manifold-modeling flow with sep. encoder | MFMF' | `--algorithm emf`



## Acknowledgements

The code is strongly based on the excellent [Neural Spline Flow code base](https://github.com/bayesiains/nsf) by C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios, see [1906.04032](https://arxiv.org/abs/1906.04032) for their paper.
