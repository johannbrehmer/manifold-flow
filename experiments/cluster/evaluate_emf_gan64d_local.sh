#!/bin/bash

conda activate ml
dir=/Users/ANONYMOUS/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

for i in 0 1 6
do
    python evaluate.py -c configs/evaluate_mf_gan64d_april.config --algorithm emf -i $i
done
