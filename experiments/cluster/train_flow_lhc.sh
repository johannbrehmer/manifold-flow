#!/bin/bash

#SBATCH --job-name=t-sf-lhc
#SBATCH --output=log_train_flow_lhc.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
cd /scratch/jb6504/manifold-flow/experiments

#python -u train.py --modelname small --dataset lhc --algorithm flow --modellatentdim 9 --samplesize 100000 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname long --dataset lhc --algorithm flow --modellatentdim 9 --epochs 50 --dir /scratch/jb6504/manifold-flow
# python -u train.py --dataset lhc --algorithm flow --modellatentdim 9 --dir /scratch/jb6504/manifold-flow
