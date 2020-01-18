#!/bin/bash

#SBATCH --job-name=t-smf-sg
#SBATCH --output=log_train_smf_spherical.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
cd /scratch/jb6504/manifold-flow/experiments

python -u train.py --dataset spherical_gaussian --epsilon 0.01 --algorithm mf --specified --dropout 0 --dir /scratch/jb6504/manifold-flow
python -u train.py --dataset spherical_gaussian --epsilon 0.001 --algorithm mf --specified --dropout 0 --dir /scratch/jb6504/manifold-flow
python -u train.py --dataset spherical_gaussian --epsilon 0.1 --algorithm mf --specified --dropout 0 --dir /scratch/jb6504/manifold-flow
