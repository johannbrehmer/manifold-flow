#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u train.py --dataset spherical_gaussian --algorithm dough --l2reg 0.1 --doughl1reg 0.1 --dir $basedir --samplesize 1000000
