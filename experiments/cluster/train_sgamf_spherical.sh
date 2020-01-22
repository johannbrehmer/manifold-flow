#!/bin/bash

#SBATCH --job-name=t-sgamf-sg
#SBATCH --output=log_train_sgamf_spherical.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
cd /scratch/jb6504/manifold-flow/experiments

python -u train.py --modelname largebs --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.01 --dropout 0 --genbatchsize 1000 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname largebs --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.001 --dropout 0 --genbatchsize 1000 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname largebs --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.1 --dropout 0 --genbatchsize 1000 --dir /scratch/jb6504/manifold-flow
