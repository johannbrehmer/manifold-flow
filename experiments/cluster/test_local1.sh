#!/usr/bin/env bash

source activate ml
export OMP_NUM_THREADS=1
export OMP_NUM_THREADS=1
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

SLURM_ARRAY_TASK_ID=15

run=$((SLURM_ARRAY_TASK_ID / 12))
task=$((SLURM_ARRAY_TASK_ID % 12))
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}, task = ${task}, run = ${run}"

python -u generate_data.py --dataset spherical_gaussian --epsilon 0.001 --ood 10000 -i $run --dir $basedir
