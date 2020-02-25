#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u train.py --modelname small_alternate --dataset power --algorithm gamf --alternate --samplesize 100000 --dir $basedir
