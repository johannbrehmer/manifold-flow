#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

for SLURM_ARRAY_TASK_ID in 0 1 2
do
    python -u evaluate.py --modelname small_shallow_long --dataset power --algorithm flow  --outerlayers 3 --innerlayers 3 --gridresolution 101 -i ${SLURM_ARRAY_TASK_ID} --dir $basedir
    python -u evaluate.py --modelname small_long --dataset power --algorithm flow  --gridresolution 101 -i ${SLURM_ARRAY_TASK_ID} --dir $basedir
    python -u evaluate.py --modelname small --dataset power --algorithm flow --gridresolution 101 -i ${SLURM_ARRAY_TASK_ID} --dir $basedir

    python -u evaluate.py --modelname small_shallow_long --dataset power --algorithm pie  --outerlayers 3 --innerlayers 3 --gridresolution 101 -i ${SLURM_ARRAY_TASK_ID} --dir $basedir
    python -u evaluate.py --modelname small_long --dataset power --algorithm pie  --gridresolution 101 -i ${SLURM_ARRAY_TASK_ID} --dir $basedir
    python -u evaluate.py --modelname small --dataset power --algorithm pie --gridresolution 101 -i ${SLURM_ARRAY_TASK_ID} --dir $basedir
done
