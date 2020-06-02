#!/bin/bash

conda activate ml
dir=/Users/ANONYMOUS/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

for i in 0 1 2
do
    python evaluate.py -c configs/evaluate_flow_gan64d_april.config -i $i --skipgeneration
done
