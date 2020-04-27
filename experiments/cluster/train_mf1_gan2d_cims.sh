#!/bin/bash

#SBATCH --job-name=t-mf-g
#SBATCH --output=log_train_mf1_gan2d.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:4

conda activate ml
export PATH="/home/brehmer/miniconda3/envs/ml/bin/:$PATH"
export OMP_NUM_THREADS=1
dir=/home/brehmer/manifold-flow
cd $dir/experiments

python -u train.py -c cluster/configs/train_mf_gan2d_may_l1.config --modelname l1_may --algorithm mf --sequential --dir $dir
