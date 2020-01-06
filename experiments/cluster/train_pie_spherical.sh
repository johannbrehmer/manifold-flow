#!/bin/bash

#SBATCH --job-name=t-pie-sg
#SBATCH --output=log_train_pie_spherical.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
cd /scratch/jb6504/manifold-flow/experiments

# python -u train.py --dataset spherical_gaussian --algorithm pie --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
# python -u train.py --dataset spherical_gaussian --algorithm pie --epsilon 0.001 --dir /scratch/jb6504/manifold-flow
# python -u train.py --dataset spherical_gaussian --algorithm pie --epsilon 0.1 --dir /scratch/jb6504/manifold-flow

python -u train.py --modelname long --dataset spherical_gaussian --algorithm pie --epsilon 0.01 --epochs 200 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname long_shallow --dataset spherical_gaussian --algorithm pie --epsilon 0.01 --epochs 200 --outerlayers 3 --innerlayers 3 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname long_deep --dataset spherical_gaussian --algorithm pie --epsilon 0.01 --epochs 200 --outerlayers 10 --innerlayers 10 --dir /scratch/jb6504/manifold-flow
