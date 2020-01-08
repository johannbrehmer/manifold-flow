#!/bin/bash

#SBATCH --job-name=e-hy-tth
#SBATCH --output=log_evaluate_hybrid_tth.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
cd /scratch/jb6504/manifold-flow/experiments

python -u evaluate.py --dataset tth --algorithm hybrid --outercouplingmlp --outercouplinglayers 1 --outercouplinghidden 100 --modellatentdim 20 --observedsamples 1000 --dir /scratch/jb6504/manifold-flow
