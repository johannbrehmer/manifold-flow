#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u train.py --dataset tth2d --algorithm flow --modellatentdim 20 --dir $basedir --samplesize 1000000
python -u evaluate.py --dataset tth2d --algorithm flow --modellatentdim 20 --dir $basedir