#!/bin/bash

#SBATCH --job-name=t-piee-sg
#SBATCH --output=log_train_pie_epsilon_spherical.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
cd /scratch/jb6504/manifold-flow/experiments

python -u train.py --modelname pieepsilon03 --dataset spherical_gaussian --epsilon 0.01 --algorithm pie --pieepsilon 0.3 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname pieepsilon01 --dataset spherical_gaussian --epsilon 0.01 --algorithm pie --pieepsilon 0.1 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname pieepsilon003 --dataset spherical_gaussian --epsilon 0.01 --algorithm pie --pieepsilon 0.03 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname pieepsilon0003 --dataset spherical_gaussian --epsilon 0.01 --algorithm pie --pieepsilon 0.003 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname pieepsilon0001 --dataset spherical_gaussian --epsilon 0.01 --algorithm pie --pieepsilon 0.001 --dir /scratch/jb6504/manifold-flow
