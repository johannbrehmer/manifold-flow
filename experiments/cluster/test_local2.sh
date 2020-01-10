#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u evaluate.py --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm flow --modellatentdim 2 --dir $basedir
python -u evaluate.py --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm pie --modellatentdim 2 --dir $basedir
python -u evaluate.py --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm gamf --modellatentdim 2 --dir $basedir
