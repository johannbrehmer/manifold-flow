#!/bin/bash

#SBATCH --job-name=t-mf-sg
#SBATCH --output=log_train_mf_spherical_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate ml
cd /scratch/jb6504/manifold-flow/experiments

case ${SLURM_ARRAY_TASK_ID} in
0) python -u train.py --dataset spherical_gaussian --algorithm mf --outercouplingmlp --outercouplinglayers 1 --outercouplinghidden 100 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow ;;
1) python -u train.py --dataset spherical_gaussian --algorithm mf --outercouplingmlp --outercouplinglayers 1 --outercouplinghidden 100 --epsilon 0.001 --dir /scratch/jb6504/manifold-flow ;;
2) python -u train.py --dataset spherical_gaussian --algorithm mf --outercouplingmlp --outercouplinglayers 1 --outercouplinghidden 100 --epsilon 0.1 --dir /scratch/jb6504/manifold-flow ;;
*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac
