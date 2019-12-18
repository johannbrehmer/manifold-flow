#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u train.py --dataset tth --algorithm gamf --modellatentdim 20 --epochs 12 --batchsize 200 --samplesize 100000 --debug --dir $basedir
python -u evaluate.py --dataset tth --algorithm gamf --modellatentdim 20 --debug --dir $basedir
