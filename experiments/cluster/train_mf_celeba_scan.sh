#!/bin/bash

#SBATCH --job-name=t-mf-c
#SBATCH --output=log_train_mf_celeba_scan_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

nvcc --version
nvidia-smi

dim=$((2**$SLURM_ARRAY_TASK_ID))
python -u train.py -c configs/train_mf_celeba_scan.config -i 0 --modelname scan_${dim} --modellatentdim ${dim}
