#!/bin/bash

#SBATCH --job-name=e-gamf-lhc
#SBATCH --output=log_evaluate_gamf_lhc.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
cd /scratch/jb6504/manifold-flow/experiments

python -u evaluate.py --modelname small_largebs --dataset lhc --algorithm gamf --modellatentdim 20 --samplesize 100000 --dir /scratch/jb6504/manifold-flow
