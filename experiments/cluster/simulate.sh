#!/bin/bash

#SBATCH --job-name=sim
#SBATCH --output=log_simulate_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/ANONYMOUS/manifold-flow/experiments

python -u generate_data.py --dataset power --ood 10000 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/ANONYMOUS/manifold-flow
