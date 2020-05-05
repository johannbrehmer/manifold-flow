#!/usr/bin/env bash

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

python -u train.py -c cluster/configs/train_mfmf_gan2d_april.config --modelname debug --algorithm mf --debug --batchsize 100 --samplesize 1000 --epochs 10 --dir $dir
