#!/bin/bash

#SBATCH --job-name=aef-t-tth
#SBATCH --output=log_train_latent.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate madminer
cd /scratch/jb6504/autoencoded-flow/

python -u evaluate.py tth tth_latent_{} --dir /scratch/jb6504/autoencoded-flow
