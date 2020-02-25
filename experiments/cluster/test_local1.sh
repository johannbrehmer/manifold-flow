#!/usr/bin/env bash

source activate ml
export OMP_NUM_THREADS=1
export SLURM_ARRAY_TASK_ID=0
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

# python -u generate_data.py --dataset power --ood 10000 -i 0 --dir $basedir
python -u evaluate.py --truth --dataset power -i 0 --dir $basedir
