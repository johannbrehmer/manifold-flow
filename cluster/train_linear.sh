#!/bin/bash

#SBATCH --job-name=aef-t
#SBATCH --output=log_train_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate madminer
cd /scratch/jb6504/autoencoded-flow/

# python -u train.py tth_latent_${SLURM_ARRAY_TASK_ID} --latent ${SLURM_ARRAY_TASK_ID} --alpha 0.001 --dir /scratch/jb6504/autoencoded-flow

python -u train.py gaussian_8_8_${SLURM_ARRAY_TASK_ID} --latent ${SLURM_ARRAY_TASK_ID} --dataset gaussian -x 8 --dir /scratch/jb6504/autoencoded-flow
python -u train.py gaussian_8_16_${SLURM_ARRAY_TASK_ID} --latent ${SLURM_ARRAY_TASK_ID} --dataset gaussian -x 16 --dir /scratch/jb6504/autoencoded-flow
python -u train.py gaussian_8_32_${SLURM_ARRAY_TASK_ID} --latent ${SLURM_ARRAY_TASK_ID} --dataset gaussian -x 32 --dir /scratch/jb6504/autoencoded-flow
python -u train.py gaussian_8_64_${SLURM_ARRAY_TASK_ID} --latent ${SLURM_ARRAY_TASK_ID} --dataset gaussian -x 64 --dir /scratch/jb6504/autoencoded-flow
python -u train.py gaussian_8_128_${SLURM_ARRAY_TASK_ID} --latent ${SLURM_ARRAY_TASK_ID} --dataset gaussian -x 128 --dir /scratch/jb6504/autoencoded-flow
