#!/usr/bin/env bash

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

python -u train.py -c cluster/configs/train_lhc_may.config --dataset lhc40d --modelname scandal_debug --algorithm flow --scandal 1. --samplesize 1000 --dir $dir
