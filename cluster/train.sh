#!/bin/bash

#SBATCH --job-name=aef-t-tth-%a
#SBATCH --output=log_train_latent_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate madminer
cd /scratch/jb6504/autoencoded-flow/

python -u train.py tth_latent_${SLURM_ARRAY_TASK_ID} --latent ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/autoencoded-flow
