#!/bin/bash

#SBATCH --job-name=e-t-p
#SBATCH --output=log_evaluate_truth_power_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/ANONYMOUS/manifold-flow/experiments

python -u evaluate.py --truth --dataset power --gridresolution 101 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/ANONYMOUS/manifold-flow
