#!/usr/bin/env bash

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

python -u train.py -c cluster/configs/train_flow_gan2d_april.config --algorithm flow --debug --batchsize 50 --samplesize 1000 --dir $dir
