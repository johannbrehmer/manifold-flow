#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u evaluate.py --modelname march --dataset lhc --algorithm flow --modellatentdim 14 --splinebins 10 --observedsamples 100 -i 0 --dir $basedir
