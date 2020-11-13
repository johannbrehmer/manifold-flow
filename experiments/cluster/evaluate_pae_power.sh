#!/bin/bash

#SBATCH --job-name=e-pae-p
#SBATCH --output=log_evaluate_pae_power_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u evaluate.py -c configs/evaluate_power_march.config --modelname sequential_march --algorithm pae -i ${SLURM_ARRAY_TASK_ID}
