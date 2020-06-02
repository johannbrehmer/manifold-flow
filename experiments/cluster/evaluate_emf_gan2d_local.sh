#!/bin/bash

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

for i in 9
do
    python evaluate.py -c configs/evaluate_mf_gan2d_april.config --algorithm emf -i $i
done
