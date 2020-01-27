#!/bin/bash

#SBATCH --job-name=t-pie-lhc
#SBATCH --output=log_train_pie_lhc.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
cd /scratch/jb6504/manifold-flow/experiments

python -u train.py --modelname small --dataset lhc --algorithm pie --modellatentdim 9 --samplesize 100000 --dir /scratch/jb6504/manifold-flow
