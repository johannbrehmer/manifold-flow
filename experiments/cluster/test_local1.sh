#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

for i in 0 1 2
do
    python -u evaluate.py --truth --dataset power --gridresolution 101 -i $i --dir $basedir
done

for i in 0 1 2
do
    python -u evaluate.py --modelname small_shallow_long --dataset power --algorithm flow  --outerlayers 3 --innerlayers 3 --gridresolution 101 -i $i --dir $basedir
    python -u evaluate.py --modelname small_long --dataset power --algorithm flow --gridresolution 101  -i $i --dir $basedir
    python -u evaluate.py --modelname small --dataset power --algorithm flow --gridresolution 101 -i $i --dir $basedir
done
