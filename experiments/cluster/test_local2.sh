#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u train.py --modelname debug --dataset power --algorithm mf --alternate --samplesize 100000 --weightdecay 1.e-5 --debug --dir $basedir
