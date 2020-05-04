#!/usr/bin/env bash

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

python -u train.py -c cluster/configs/train_mfmf_gan2d_april.config --modelname debug --algorithm mf --debug --batchsize 50 --samplesize 200 --epochs 2 --dir $dir
