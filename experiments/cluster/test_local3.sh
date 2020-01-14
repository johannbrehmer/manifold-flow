#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u train.py --dataset spherical_gaussian --epsilon 0.01 --algorithm gamf --epochs 4 --samplesize 10000 --specified --dir $basedir
