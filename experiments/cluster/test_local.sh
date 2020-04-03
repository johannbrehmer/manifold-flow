#!/usr/bin/env bash

source activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments


python -u evaluate.py -c cluster/configs/evaluate_lhc_may.config --dataset lhc2d --modellatentdim 2 --modelname may --algorithm flow --dir $dir
