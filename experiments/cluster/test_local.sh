#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u paramscan.py --samplesize 10000 --paramscanstudyname debug --dir $basedir
