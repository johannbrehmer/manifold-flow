#!/usr/bin/env bash

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

python -u train.py -c cluster/configs/train_mfmf_gan2d_april.config --algorithm mf --sequential --debug --batchsize 50 --epochs 200 --dir $dir
