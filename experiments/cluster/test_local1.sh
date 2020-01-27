#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u train.py --modelname debug --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.001 --genbatchsize 100 --samplesize 10000 --dir $basedir
