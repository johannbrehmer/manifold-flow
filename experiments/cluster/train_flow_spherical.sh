#!/bin/bash

#SBATCH --job-name=mf-t-sf-sg
#SBATCH --output=log_train_flow_spherical.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate ml
cd /scratch/jb6504/manifold-flow/experiments

python -u train.py --algorithm flow --datadim 15 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow

python -u train.py --algorithm flow --datadim 15 --epsilon 0.001 --dir /scratch/jb6504/manifold-flow
python -u train.py --algorithm flow --datadim 15 --epsilon 0.003 --dir /scratch/jb6504/manifold-flow
python -u train.py --algorithm flow --datadim 15 --epsilon 0.03 --dir /scratch/jb6504/manifold-flow
python -u train.py --algorithm flow --datadim 15 --epsilon 0.1 --dir /scratch/jb6504/manifold-flow

python -u train.py --algorithm flow --datadim 11 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
python -u train.py --algorithm flow --datadim 13 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
python -u train.py --algorithm flow --datadim 20 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
python -u train.py --algorithm flow --datadim 30 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
