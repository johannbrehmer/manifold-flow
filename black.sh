#!/usr/bin/env bash

source activate ml

black -l 180 experiments/*.py experiments/evaluation/*.py experiments/datasets/*.py experiments/training/*.py experiments/architectures/*.py
black -l 180 manifold_flow/distributions/*.py manifold_flow/flows/*.py manifold_flow/nn/*.py manifold_flow/transforms/*.py manifold_flow/transforms/splines/*.py manifold_flow/utils/*.py
