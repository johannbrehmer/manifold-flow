#!/bin/bash

#SBATCH --job-name=t-f-g
#SBATCH --output=log_train_flow_gan2d.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_12gb

conda activate ml
export PATH="/home/brehmer/miniconda3/envs/ml/bin/:$PATH"
export OMP_NUM_THREADS=1
dir=/home/brehmer/manifold-flow
cd $dir/experiments

python -u train.py -c cluster/configs/train_flow_gan2d_april.config --modelname april --algorithm flow --dir $dir
