#!/bin/bash

#SBATCH --job-name=t-sgamf-sg
#SBATCH --output=log_train_sgamf_spherical.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u train.py --modelname small_hugebs --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.01 --genbatchsize 5000 --epochs 1000 --samplesize 100000 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname small_hugebs --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.001 --genbatchsize 5000 --epochs 1000 --samplesize 100000 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname small_hugebs --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.1 --genbatchsize 5000 --epochs 1000 --samplesize 100000 --dir /scratch/jb6504/manifold-flow

python -u train.py --modelname small_largebs_shallow_long --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.01 --samplesize 100000 --genbatchsize 1000 --epochs 500 --outerlayers 3 --innerlayers 3 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname small_largebs_shallow_long --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.001 --samplesize 100000 --genbatchsize 1000 --epochs 500 --outerlayers 3 --innerlayers 3 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname small_largebs_shallow_long --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.1 --samplesize 100000 --genbatchsize 1000 --epochs 500 --outerlayers 3 --innerlayers 3 --dir /scratch/jb6504/manifold-flow

python -u train.py --modelname small_largebs_long --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.01 --samplesize 100000 --genbatchsize 1000 --epochs 500 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname small_largebs_long --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.001 --samplesize 100000 --genbatchsize 1000 --epochs 500 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname small_largebs_long --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.1 --samplesize 100000 --genbatchsize 1000 --epochs 500 --dir /scratch/jb6504/manifold-flow

#python -u train.py --modelname small_largebs --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.01 --genbatchsize 1000 --epochs 200 --samplesize 100000 --dir /scratch/jb6504/manifold-flow
#python -u train.py --modelname small_largebs --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.001 --genbatchsize 1000 --epochs 200 --samplesize 100000 --dir /scratch/jb6504/manifold-flow
#python -u train.py --modelname small_largebs --dataset spherical_gaussian --algorithm gamf --specified --epsilon 0.1 --genbatchsize 1000 --epochs 200 --samplesize 100000 --dir /scratch/jb6504/manifold-flow
