#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --constraint=gpu_12gb
#SBATCH --job-name=t-mfs-c
#SBATCH --output=log_train_mf_small_celeba_%a.log

module load cuda-10.2
conda activate ml
export PATH="/home/ANONYMOUS/miniconda3/envs/ml/bin/:$PATH"
export OMP_NUM_THREADS=1
dir=/data/ANONYMOUS/manifold-flow
cd $dir/experiments

nvcc --version
nvidia-smi

python -u train.py -c configs/train_mf_celeba_april.config --modellatentdim 128 -i ${SLURM_ARRAY_TASK_ID} --dir $dir
