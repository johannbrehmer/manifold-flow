#!/bin/bash

#SBATCH --job-name=t-mf-g
#SBATCH --output=log_train_mf_gan64d_%a.log
#SBATCH -p usatlas
#SBATCH -t 1-00:00:00
#SBATCH --qos=usatlas
#SBATCH --account=tier3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

source ~/.bashrc
module load cuda/10.1
conda activate ml
export OMP_NUM_THREADS=1
cd /sdcc/u/ANONYMOUS/manifold-flow/experiments

nvcc --version
nvidia-smi

python -u train.py -c configs/train_mf_gan64d_april.config -i ${SLURM_ARRAY_TASK_ID} --dir /sdcc/u/ANONYMOUS/manifold-flow
