#!/usr/bin/env bash

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

python -u train.py -c cluster/configs/train_flow_celeba_april.config --algorithm mf --sequential --debug --dir $dir
