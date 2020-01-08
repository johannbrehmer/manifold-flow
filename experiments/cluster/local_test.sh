#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

# python -u generate_data.py --dataset conditional_spherical_gaussian --epsilon 0.01
python -u train.py --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm pie --modellatentdim 2 --epochs 10 --samplesize 100000 --debug --dir $basedir
python -u evaluate.py --dataset conditional_spherical_gaussian  --epsilon 0.01 --algorithm pie --modellatentdim 2 --thin 1 --debug --dir $basedir

# python -u evaluate.py --dataset tth --algorithm flow --modellatentdim 20 --thin 1 --debug --dir $basedir --observedsamples 1000
