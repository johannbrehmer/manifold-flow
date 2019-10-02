#!/bin/bash

#SBATCH --job-name=manifold_flow-e
#SBATCH --output=log_eval_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate madminer
cd /scratch/jb6504/autoencoded-flow/

python -u evaluate.py gaussian_${SLURM_ARRAY_TASK_ID} --data ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/autoencoded-flow
