#!/bin/bash

#SBATCH --job-name=manifold_flow-t-sph
#SBATCH --output=log_train_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate madminer
cd /scratch/jb6504/autoencoded-flow/

# python -u train.py tth_latent_${SLURM_ARRAY_TASK_ID} --latent ${SLURM_ARRAY_TASK_ID} --alpha 0.001 --dir /scratch/jb6504/autoencoded-flow

python -u train.py spherical_gaussian_15_16_${SLURM_ARRAY_TASK_ID} --latent ${SLURM_ARRAY_TASK_ID} --dataset spherical_gaussian -x 16 --dir /scratch/jb6504/autoencoded-flow
python -u train.py spherical_gaussian_15_32_${SLURM_ARRAY_TASK_ID} --latent ${SLURM_ARRAY_TASK_ID} --dataset spherical_gaussian -x 32 --dir /scratch/jb6504/autoencoded-flow
python -u train.py spherical_gaussian_15_64_${SLURM_ARRAY_TASK_ID} --latent ${SLURM_ARRAY_TASK_ID} --dataset spherical_gaussian -x 64 --dir /scratch/jb6504/autoencoded-flow
python -u train.py spherical_gaussian_15_128_${SLURM_ARRAY_TASK_ID} --latent ${SLURM_ARRAY_TASK_ID} --dataset spherical_gaussian -x 128 --dir /scratch/jb6504/autoencoded-flow
