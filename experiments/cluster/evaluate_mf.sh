#!/bin/bash

#SBATCH --job-name=mf-e-mf-sg
#SBATCH --output=log_evaluate_mf_spherical.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
cd /scratch/jb6504/manifold-flow/experiments

python -u evaluate.py --algorithm mf --datadim 15 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow

python -u evaluate.py --algorithm mf --datadim 15 --epsilon 0.001 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --algorithm mf --datadim 15 --epsilon 0.003 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --algorithm mf --datadim 15 --epsilon 0.03 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --algorithm mf --datadim 15 --epsilon 0.1 --dir /scratch/jb6504/manifold-flow

python -u evaluate.py --algorithm mf --datadim 11 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --algorithm mf --datadim 13 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --algorithm mf --datadim 20 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --algorithm mf --datadim 30 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
