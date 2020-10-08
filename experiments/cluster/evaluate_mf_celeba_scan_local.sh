#!/bin/bash

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

for i in 10
do
    echo ""
    dim=$((2**$i))
    echo "Starting evaluation for n = $dim"
    python -u evaluate.py -c configs/evaluate_mf_celeba_scan.config -i 0 --modelname scan_${dim} --modellatentdim ${dim}
    echo ""
    echo ""
done
