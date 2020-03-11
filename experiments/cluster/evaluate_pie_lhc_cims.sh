#!/bin/bash

#SBATCH --job-name=e-pie-lhc
#SBATCH --output=log_evaluate_pie_lhc_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB
#SBATCH --time=2-00:00:00
# #SBATCH --gres=gpu:1

conda activate ml
export PATH="/home/brehmer/miniconda3/envs/ml/bin/:$PATH"
export OMP_NUM_THREADS=1
dir=/home/brehmer/manifold-flow

cd $dir/experiments
python -u evaluate.py --modelname march --dataset lhc --algorithm pie --modellatentdim 14 --splinebins 10 --observedsamples 100 -i ${SLURM_ARRAY_TASK_ID} --dir $dir
