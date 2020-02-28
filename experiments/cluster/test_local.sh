#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u train.py --modelname small_alternate --dataset power --algorithm mf --alternate --samplesize 100000 --debug --weightdecay 1.e-5 --dir $basedir
# python -u evaluate.py --modelname small_alternate --dataset power --algorithm mf --gridresolution 101 --dir $basedir
