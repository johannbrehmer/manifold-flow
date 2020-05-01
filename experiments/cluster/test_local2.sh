#!/usr/bin/env bash

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

for i in 3 4
do
    python -u evaluate.py -c cluster/configs/evaluate_power_march.config --modelname conditionalmanifold_march --algorithm pie --conditionalouter -i $i --dir $dir
done
