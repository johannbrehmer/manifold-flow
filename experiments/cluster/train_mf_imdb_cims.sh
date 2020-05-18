#!/bin/bash

#SBATCH --job-name=t-mf-i
#SBATCH --output=log_train_mf_imdb_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --constraint=gpu_12gb

module load cuda-10.2
conda activate ml
export PATH="/home/brehmer/miniconda3/envs/ml/bin/:$PATH"
export OMP_NUM_THREADS=1
dir=/data/brehmer/manifold-flow
cd $dir/experiments

nvcc --version
nvidia-smi

python -u train.py -c configs/train_mf_imdb_april.config --resume 15 -i ${SLURM_ARRAY_TASK_ID}  # --resume 16 for run 1
