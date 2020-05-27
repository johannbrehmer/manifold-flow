#!/bin/bash

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

for i in 0 1 2
do
    # python evaluate.py -c configs/evaluate_mf_celeba_april.config --algorithm emf -i $i
    python evaluate.py -c configs/evaluate_mf_celeba_april.config --algorithm emf --modelname long_april -i $i
done
