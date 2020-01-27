#!/bin/bash

#SBATCH --job-name=e-mf-lhc
#SBATCH --output=log_evaluate_mf_lhc.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
cd /scratch/jb6504/manifold-flow/experiments

python -u evaluate.py --modelname small --dataset lhc --algorithm mf --outercouplingmlp --outercouplinglayers 1 --modellatentdim 9 --samplesize 100000 --dir /scratch/jb6504/manifold-flow
