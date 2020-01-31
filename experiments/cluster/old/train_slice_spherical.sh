#!/bin/bash

#SBATCH --job-name=t-sl-sg
#SBATCH --output=log_train_slice_spherical.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u train.py --dataset spherical_gaussian --algorithm slice --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
python -u train.py --dataset spherical_gaussian --algorithm slice --epsilon 0.001 --dir /scratch/jb6504/manifold-flow
python -u train.py --dataset spherical_gaussian --algorithm slice --epsilon 0.1 --dir /scratch/jb6504/manifold-flow
