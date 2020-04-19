#!/bin/bash

#SBATCH --job-name=t-pie-c
#SBATCH --output=log_train_pie_celeba.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:4

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/image-mf/experiments

python -u train.py -c cluster/configs/train_flow_celeba_may.config --algorithm pie
