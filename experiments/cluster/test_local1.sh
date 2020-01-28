#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u train.py --modelname debug --dataset spherical_gaussian --algorithm emf --epsilon 0.01 --samplesize 10000 --dir $basedir
