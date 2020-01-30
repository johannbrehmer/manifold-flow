#!/bin/bash

#SBATCH --job-name=e-sgamf-csg
#SBATCH --output=log_evaluate_sgamf_csg.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
cd /scratch/jb6504/manifold-flow/experiments


python -u evaluate.py --modelname small_hugebs --dataset conditional_spherical_gaussian --algorithm gamf --specified --epsilon 0.01  --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_hugebs --dataset conditional_spherical_gaussian --algorithm gamf --specified --epsilon 0.001  --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_hugebs --dataset conditional_spherical_gaussian --algorithm gamf --specified --epsilon 0.1  --dir /scratch/jb6504/manifold-flow

python -u evaluate.py --modelname small_largebs_shallow_long --dataset conditional_spherical_gaussian --algorithm gamf --specified --epsilon 0.01  --outerlayers 3 --innerlayers 3 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_largebs_shallow_long --dataset conditional_spherical_gaussian --algorithm gamf --specified --epsilon 0.001  --outerlayers 3 --innerlayers 3 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_largebs_shallow_long --dataset conditional_spherical_gaussian --algorithm gamf --specified --epsilon 0.1  --outerlayers 3 --innerlayers 3 --dir /scratch/jb6504/manifold-flow

python -u evaluate.py --modelname small_largebs_long --dataset conditional_spherical_gaussian --algorithm gamf --specified --epsilon 0.01  --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_largebs_long --dataset conditional_spherical_gaussian --algorithm gamf --specified --epsilon 0.001  --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_largebs_long --dataset conditional_spherical_gaussian --algorithm gamf --specified --epsilon 0.1  --dir /scratch/jb6504/manifold-flow

python -u evaluate.py --modelname small_largebs --dataset conditional_spherical_gaussian --algorithm gamf --specified --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_largebs --dataset conditional_spherical_gaussian --algorithm gamf --specified --epsilon 0.001 --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_largebs --dataset conditional_spherical_gaussian --algorithm gamf --specified --epsilon 0.1 --dir /scratch/jb6504/manifold-flow

#python -u evaluate.py --modelname largebs --dataset conditional_spherical_gaussian --algorithm gamf --specified --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
#python -u evaluate.py --modelname largebs --dataset conditional_spherical_gaussian --algorithm gamf --specified --epsilon 0.001 --dir /scratch/jb6504/manifold-flow
#python -u evaluate.py --modelname largebs --dataset conditional_spherical_gaussian --algorithm gamf --specified --epsilon 0.1 --dir /scratch/jb6504/manifold-flow
