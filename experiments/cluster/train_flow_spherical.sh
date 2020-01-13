#!/bin/bash

#SBATCH --job-name=t-sf-sg
#SBATCH --output=log_train_flow_spherical.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
cd /scratch/jb6504/manifold-flow/experiments

python -u train.py --dataset spherical_gaussian --algorithm flow --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
# python -u train.py --dataset spherical_gaussian --algorithm flow --epsilon 0.001 --dir /scratch/jb6504/manifold-flow
# python -u train.py --dataset spherical_gaussian --algorithm flow --epsilon 0.1 --dir /scratch/jb6504/manifold-flow

python -u train.py --modelname reg0001 --dataset spherical_gaussian --algorithm flow --epsilon 0.01 --l2reg 0.001 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname reg001 --dataset spherical_gaussian --algorithm flow --epsilon 0.01 --l2reg 0.01 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname reg01 --dataset spherical_gaussian --algorithm flow --epsilon 0.01 --l2reg 0.1 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname reg1 --dataset spherical_gaussian --algorithm flow --epsilon 0.01 --l2reg 1.0 --dir /scratch/jb6504/manifold-flow
