#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u evaluate.py --truth --dataset power --gridresolution 101 --dir /scratch/jb6504/manifold-flow
