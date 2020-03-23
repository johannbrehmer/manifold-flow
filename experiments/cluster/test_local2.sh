#!/usr/bin/env bash

source activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

python -u train.py --modelname debug --dataset lhc40d --algorithm mf --alternate --modellatentdim 14 --splinebins 10 --samplesize 10000 --nllfactor 0.1 --subsets 100 -i 0 --dir $dir

