#!/usr/bin/env bash

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

python -u paramscan.py -c cluster/configs/paramscan_lhc_may.config --samplesize 100000 --dir $dir
