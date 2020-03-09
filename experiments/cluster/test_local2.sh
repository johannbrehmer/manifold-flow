#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

for i in 0 1 2
do
    # python -u evaluate.py --modelname march --dataset lhc2d --algorithm flow --modellatentdim 2 --observedsamples 100 -i $i --dir $basedir
    python -u evaluate.py --modelname march --dataset lhc --algorithm pie --modellatentdim 14 --splinebins 10 --observedsamples 100 -i $i --dir $basedir
    # python -u evaluate.py --modelname march --dataset lhc --algorithm flow --modellatentdim 14 --splinebins 10 --observedsamples 100 -i $i --dir $basedir
done
