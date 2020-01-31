#!/bin/bash

#SBATCH --job-name=t-sf-lhc
#SBATCH --output=log_train_flow_lhc.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u train.py --dataset lhc --algorithm flow --modellatentdim 9 --epochs 100 --dir /scratch/jb6504/manifold-flow
