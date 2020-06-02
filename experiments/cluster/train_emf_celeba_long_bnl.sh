#!/bin/bash

#SBATCH -p usatlas
#SBATCH -t 1-00:00:00
#SBATCH --qos=usatlas
#SBATCH --account=tier3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=t-emfl-c
#SBATCH --output=log_train_emf_long_celeba_%a.log

source ~/.bashrc
module load cuda/10.1
conda activate ml
export OMP_NUM_THREADS=1
cd /sdcc/u/brehmer/manifold-flow/experiments

nvcc --version
nvidia-smi

python -u train.py -c configs/train_mf_celeba_april.config --modelname long_april --epochs 200 --resume 100 --algorithm emf -i ${SLURM_ARRAY_TASK_ID}
