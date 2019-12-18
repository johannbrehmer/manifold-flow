#!/bin/bash

#SBATCH --job-name=t-sf-tth
#SBATCH --output=log_train_flow_tth.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
cd /scratch/jb6504/manifold-flow/experiments

python -u train.py --dataset tth --algorithm flow --modellatentdim 20 --dir /scratch/jb6504/manifold-flow
