#!/bin/bash

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

python evaluate.py -c cluster/configs/evaluate_mf_gan2d_april.config
