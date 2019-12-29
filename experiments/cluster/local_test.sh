#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u generate_data.py --dataset conditional_spherical_gaussian --epsilon 0.001
python -u train.py --dataset conditional_spherical_gaussian --algorithm pie --specified --modellatentdim 2 --epochs 12 --batchsize 200 --samplesize 10000 --epsilon 0.001 --debug --dir $basedir
# python -u evaluate.py --dataset tth --algorithm hybrid --modellatentdim 20 --thin 1 --debug --dir $basedir
