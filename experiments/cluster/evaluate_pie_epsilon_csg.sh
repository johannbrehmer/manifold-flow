#!/bin/bash

#SBATCH --job-name=e-piee-csg
#SBATCH --output=log_evaluate_pie_epsilon_csg.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u evaluate.py --modelname small_pieepsilon03 --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm pie --pieepsilon 0.3 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_pieepsilon01 --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm pie --pieepsilon 0.1 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_pieepsilon003 --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm pie --pieepsilon 0.03 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_pieepsilon0003 --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm pie --pieepsilon 0.003 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_pieepsilon0001 --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm pie --pieepsilon 0.001 --dir /scratch/jb6504/manifold-flow
