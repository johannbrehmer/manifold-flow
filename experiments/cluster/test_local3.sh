#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

i=0

python -u evaluate.py --modelname march --dataset power --algorithm mf --splinebins 10 --splinerange 6. --gridresolution 101 --skipmcmc -i $i --dir $basedir
python -u evaluate.py --modelname alternate_march --dataset power --algorithm emf --splinebins 10 --splinerange 6. --gridresolution 101 --skipmcmc  -i $i --dir $basedir
python -u evaluate.py --modelname march --dataset power --algorithm emf --splinebins 10 --splinerange 6. --gridresolution 101  --skipmcmc -i $i --dir $basedir
