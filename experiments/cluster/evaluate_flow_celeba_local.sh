#!/bin/bash

conda activate ml
dir=/Users/ANONYMOUS/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

for i in 0 1
do
    # python evaluate.py -c configs/evaluate_flow_celeba_april.config -i $i
    python evaluate.py -c configs/evaluate_flow_celeba_april.config --modelname long_april -i $i
done
