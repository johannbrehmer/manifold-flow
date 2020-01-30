#!/bin/bash

#SBATCH --job-name=t-gamf-lhc
#SBATCH --output=log_train_gamf_lhc3.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
cd /scratch/jb6504/manifold-flow/experiments

#python -u train.py --modelname largebs --dataset lhc --algorithm gamf --modellatentdim 9 --genbatchsize 1000 --epochs 200 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname hugebs --dataset lhc --algorithm gamf --ged --modellatentdim 9 --genbatchsize 5000 --epochs 1000 --dir /scratch/jb6504/manifold-flow

python -u train.py --modelname largebs_long --dataset lhc --algorithm gamf --ged --modellatentdim 9 --genbatchsize 1000 --epochs 500 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname hugebs_long --dataset lhc --algorithm gamf --ged --modellatentdim 9 --genbatchsize 1000 --epochs 500 --dir /scratch/jb6504/manifold-flow

#python -u train.py --modelname small_largebs --dataset lhc --algorithm gamf --modellatentdim 9 --genbatchsize 1000 --samplesize 100000 --epochs 200 --dir /scratch/jb6504/manifold-flow
