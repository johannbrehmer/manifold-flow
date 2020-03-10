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
Power-law manifold | 3 | 2 | 1 | `--dataset power`
Particle physics data | 48 | 14 | 2 | `--dataset lhc`
Particle physics data compressed to summary stats | 2 | 2 | 2 | `--dataset lhc2d`
CIFAR10 | 3 * 32 * 32 | ? | - | `--dataset cifar10`
ImageNet | 3 * 64 * 64 | ? | - | `--dataset imagenet`


### Simulating data

Necessary for the first three data sets in the table above. See [experiments/generate_data.py -h](experiments/generate_data.py).


### Training 

See [experiments/train.py -h](experiments/train.py). Note that the algorithms have different internal names from the acronyms in the paper:

Algorithm | Acronym in paper | Arguments to `train.py`
--- | --- | ---
Euclidean / standard flow | EF | `--algorithm flow`
Manifold flow | MF | `--algorithm mf --specified`
Partially invertible encoder | PIE | `--algorithm pie`
Manifold-learning flow, simultaneous training | MLF-L | `--algorithm mf`
Manifold-learning flow, alternating training schedule | MLF-A | `--algorithm mf --alternate`
Manifold-learning flow, Optimal Transport training | MLF-OT | `--algorithm gamf`
Manifold-learning flow with sep. encoder, simultaneous training | EMLF-L | `--algorithm emf`
Manifold-learning flow with sep. encoder, alternating training schedule | EMLF-A | `--algorithm emf --alternate`


### Evaluation 

See [experiments/evaluate.py -h](experiments/evaluate.py) and the notebooks in [experiments/notebooks](experiments/notebooks). Note that the algorithms have different internal names from the acronyms in the paper:

Algorithm | Acronym in paper | Arguments to `evaluate.py`
--- | --- | ---
Euclidean / standard flow | EF | `--algorithm flow`
Manifold flow | MF | `--algorithm mf --specified`
Partially invertible encoder | PIE | `--algorithm pie`
Manifold-learning flow | MLF-L, MLF-A | `--algorithm mf`
Manifold-learning flow, Optimal Transport training | MLF-OT | `--algorithm gamf`
Manifold-learning flow with sep. encoder | EMLF-L, EMLF-A | `--algorithm emf`


## Acknowledgements

The code is strongly based on the excellent [Neural Spline Flow code base](https://github.com/bayesiains/nsf) by C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios, see [1906.04032](https://arxiv.org/abs/1906.04032) for their paper.
