#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

for i in 0 1 2
do
    python -u evaluate.py --truth --dataset power --gridresolution 101 -i ${i} --dir $basedir
done
