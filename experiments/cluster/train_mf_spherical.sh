#!/bin/bash

#SBATCH --job-name=t-mf-sg
#SBATCH --output=log_train_mf_spherical_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
cd /scratch/jb6504/manifold-flow/experiments

#case ${SLURM_ARRAY_TASK_ID} in
#0) python -u train.py --modelname small --dataset spherical_gaussian --epsilon 0.01 --algorithm mf --outercouplingmlp --outercouplinglayers 1 --samplesize 100000 --dir /scratch/jb6504/manifold-flow ;;
#1) python -u train.py --modelname small --dataset spherical_gaussian --epsilon 0.001 --algorithm mf --outercouplingmlp --outercouplinglayers 1 --samplesize 100000 --dir /scratch/jb6504/manifold-flow ;;
#2) python -u train.py --modelname small --dataset spherical_gaussian --epsilon 0.1  --algorithm mf --outercouplingmlp --outercouplinglayers 1 --samplesize 100000 --dir /scratch/jb6504/manifold-flow ;;
#*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
#esac

case ${SLURM_ARRAY_TASK_ID} in
0) python -u train.py --modelname small_noprepost --dataset spherical_gaussian --epsilon 0.01 --algorithm mf --outercouplingmlp --outercouplinglayers 1 --nopretraining --noposttraining --samplesize 100000 --dir /scratch/jb6504/manifold-flow ;;
1) python -u train.py --modelname small_noprepost --dataset spherical_gaussian --epsilon 0.001 --algorithm mf --outercouplingmlp --outercouplinglayers 1 --nopretraining --noposttraining --samplesize 100000 --dir /scratch/jb6504/manifold-flow ;;
2) python -u train.py --modelname small_noprepost --dataset spherical_gaussian --epsilon 0.1  --algorithm mf --outercouplingmlp --outercouplinglayers 1 --nopretraining --noposttraining --samplesize 100000 --dir /scratch/jb6504/manifold-flow ;;
3) python -u train.py --modelname small_complex --dataset spherical_gaussian --epsilon 0.01 --algorithm mf --samplesize 100000 --dir /scratch/jb6504/manifold-flow ;;
4) python -u train.py --modelname small_complex --dataset spherical_gaussian --epsilon 0.001 --algorithm mf --samplesize 100000 --dir /scratch/jb6504/manifold-flow ;;
5) python -u train.py --modelname small_complex --dataset spherical_gaussian --epsilon 0.1  --algorithm mf --samplesize 100000 --dir /scratch/jb6504/manifold-flow ;;
*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac
