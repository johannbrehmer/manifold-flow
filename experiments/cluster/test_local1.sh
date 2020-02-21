#!/usr/bin/env bash

source activate ml
export OMP_NUM_THREADS=1
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u evaluate.py --modelname small_long --dataset spherical_gaussian --algorithm flow --epsilon 0.01  -i 0 --dir $basedir
