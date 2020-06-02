#!/bin/bash

#SBATCH --job-name=t-pie-l
#SBATCH --output=log_train_pie_lhc_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/ANONYMOUS/manifold-flow/experiments

python -u train.py -c cluster/configs/train_lhc_may.config --modelname may --algorithm pie -i ${SLURM_ARRAY_TASK_ID}
