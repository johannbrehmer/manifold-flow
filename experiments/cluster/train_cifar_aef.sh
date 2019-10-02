#!/bin/bash

#SBATCH --job-name=manifold_flow-t-cifar
#SBATCH --output=log_train_cifar_aef_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate madminer
cd /scratch/jb6504/autoencoded-flow2/

python -u train.py cifar_${SLURM_ARRAY_TASK_ID} --latent ${SLURM_ARRAY_TASK_ID} --dataset cifar --dir /scratch/jb6504/autoencoded-flow2 --outer 3 --inner 10 --lr 1.e-4 --debug
