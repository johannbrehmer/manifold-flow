#!/bin/bash

#SBATCH --job-name=t-emf-g
#SBATCH --output=log_train_emf_gan2d_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_12gb

module load cuda-10.2
conda activate ml
export PATH="/home/brehmer/miniconda3/envs/ml/bin/:$PATH"
export OMP_NUM_THREADS=1
dir=/data/brehmer/manifold-flow
cd $dir/experiments

nvcc --version
which python

python -u train.py -c cluster/configs/train_mf_gan2d_april.config --algorithm emf -i ${SLURM_ARRAY_TASK_ID}
