#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u evaluate.py --truth --dataset power --gridresolution 101 --dir $basedir
python -u evaluate.py --truth --dataset power --gridresolution 101 -i 1 --dir $basedir
python -u evaluate.py --truth --dataset power --gridresolution 101 -i 2 --dir $basedir
