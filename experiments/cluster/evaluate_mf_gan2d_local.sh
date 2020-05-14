#!/bin/bash

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

for i in 0 1 2 3 4
do
    python evaluate.py -c configs/evaluate_mf_gan2d_april.config -i $i
done
