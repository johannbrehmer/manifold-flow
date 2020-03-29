#!/usr/bin/env bash

source activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

python -u evaluate.py --modelname sequential_march --dataset power --algorithm emf --splinebins 10 --splinerange 6. --gridresolution 101 -i 4 --dir $dir
python -u evaluate.py --modelname sequential_march --dataset power --algorithm mf --splinebins 10 --splinerange 6. --gridresolution 101 -i 4 --dir $dir
