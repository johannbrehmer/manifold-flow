#!/bin/bash

#SBATCH --job-name=aef-t-cifar
#SBATCH --output=log_train_cifar_flow.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate madminer
cd /scratch/jb6504/autoencoded-flow/

python -u train.py cifar_${SLURM_ARRAY_TASK_ID} --latent ${SLURM_ARRAY_TASK_ID} --dataset cifar --dir /scratch/jb6504/autoencoded-flow --outer 3 --inner 10 --debug
