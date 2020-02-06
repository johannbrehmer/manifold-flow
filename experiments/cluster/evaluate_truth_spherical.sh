#!/bin/bash

#SBATCH --job-name=e-t-sg
#SBATCH --output=log_evaluate_truth_spherical_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u evaluate.py --truth --dataset spherical_gaussian --epsilon 0.01 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --truth --dataset spherical_gaussian --epsilon 0.001 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --truth --dataset spherical_gaussian --epsilon 0.1 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
