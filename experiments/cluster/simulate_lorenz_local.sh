#!/bin/bash

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

python -u generate_data.py --dataset lorenz --train 1000000 --test 10000 --dir $dir
