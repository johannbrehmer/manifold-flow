#!/bin/bash

#SBATCH --job-name=e-sf-sg
#SBATCH --output=log_evaluate_flow_spherical.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
cd /scratch/jb6504/manifold-flow/experiments

python -u evaluate.py --modelname small --dataset spherical_gaussian --algorithm flow --epsilon 0.01 --dropout 0 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small --dataset spherical_gaussian --algorithm flow --epsilon 0.001 --dropout 0 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small --dataset spherical_gaussian --algorithm flow --epsilon 0.1 --dropout 0 --dir /scratch/jb6504/manifold-flow

python -u evaluate.py --dataset spherical_gaussian --algorithm flow --epsilon 0.01 --dropout 0 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --dataset spherical_gaussian --algorithm flow --epsilon 0.001 --dropout 0 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --dataset spherical_gaussian --algorithm flow --epsilon 0.1 --dropout 0 --dir /scratch/jb6504/manifold-flow
