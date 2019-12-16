#!/usr/bin/env bash

source activate ml

black -l 160 experiments/*.py experiments/inference/*.py experiments/simulators/*.py experiments/utils/*.py
black -l 160 manifold_flow/distributions/*.py manifold_flow/flows/*.py manifold_flow/nn/*.py manifold_flow/training/*.py manifold_flow/transforms/*.py manifold_flow/transforms/splines/*.py manifold_flow/utils/*.py
