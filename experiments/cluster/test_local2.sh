#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u train.py --dataset spherical_gaussian --algorithm mf --specified --dir $basedir --debug --samplesize 100000 --epochs 4
