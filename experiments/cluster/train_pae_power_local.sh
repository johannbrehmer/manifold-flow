#!/bin/bash

conda activate ml
cd /Users/johannbrehmer/work/projects/manifold_flow/manifold-flow/experiments

python -u train.py -c configs/train_power_march.config --modelname sequential_march --algorithm pae --sequential -i 0 --dir /Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
