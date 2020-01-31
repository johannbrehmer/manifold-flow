#!/bin/bash

#SBATCH --job-name=t-emf-sg
#SBATCH --output=log_train_emf_spherical2_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
cd /scratch/jb6504/manifold-flow/experiments

case ${SLURM_ARRAY_TASK_ID} in
0) python -u train.py --modelname small --dataset spherical_gaussian --epsilon 0.01 --algorithm emf --outercouplingmlp --outercouplinglayers 1 --samplesize 100000 --dir /scratch/jb6504/manifold-flow ;;
1) python -u train.py --modelname small --dataset spherical_gaussian --epsilon 0.001 --algorithm emf --outercouplingmlp --outercouplinglayers 1 --samplesize 100000 --dir /scratch/jb6504/manifold-flow ;;
2) python -u train.py --modelname small --dataset spherical_gaussian --epsilon 0.1  --algorithm emf --outercouplingmlp --outercouplinglayers 1 --samplesize 100000 --dir /scratch/jb6504/manifold-flow ;;

#0) python -u train.py --modelname small_shallow_long --dataset spherical_gaussian --algorithm emf --epsilon 0.01 --samplesize 100000 --epochs 50 --outerlayers 3 --innerlayers 3 --dir /scratch/jb6504/manifold-flow ;;
#1) python -u train.py --modelname small_shallow_long --dataset spherical_gaussian --algorithm emf --epsilon 0.001 --samplesize 100000 --epochs 50 --outerlayers 3 --innerlayers 3 --dir /scratch/jb6504/manifold-flow ;;
#2) python -u train.py --modelname small_shallow_long --dataset spherical_gaussian --algorithm emf --epsilon 0.1 --samplesize 100000 --epochs 50 --outerlayers 3 --innerlayers 3 --dir /scratch/jb6504/manifold-flow ;;
#
#3) python -u train.py --modelname small_long --dataset spherical_gaussian --algorithm emf --epsilon 0.01 --samplesize 100000 --epochs 50 --dir /scratch/jb6504/manifold-flow ;;
#4) python -u train.py --modelname small_long --dataset spherical_gaussian --algorithm emf --epsilon 0.001 --samplesize 100000 --epochs 50 --dir /scratch/jb6504/manifold-flow ;;
#5) python -u train.py --modelname small_long --dataset spherical_gaussian --algorithm emf --epsilon 0.1 --samplesize 100000 --epochs 50 --dir /scratch/jb6504/manifold-flow ;;

*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac
