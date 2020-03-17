#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u evaluate.py --modelname march --dataset power --algorithm flow --splinebins 10 --splinerange 6. --gridresolution 101 -i 3 --dir $basedir
python -u evaluate.py --modelname march --dataset power --algorithm pie --splinebins 10 --splinerange 6. --gridresolution 101 -i 3 --dir $basedir
python -u evaluate.py --modelname march --dataset power --algorithm flow --splinebins 10 --splinerange 6. --gridresolution 101 -i 4 --dir $basedir
python -u evaluate.py --modelname march --dataset power --algorithm pie --splinebins 10 --splinerange 6. --gridresolution 101 -i 4 --dir $basedir