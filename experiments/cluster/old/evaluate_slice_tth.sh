#!/bin/bash

#SBATCH --job-name=e-sl-lhc
#SBATCH --output=log_evaluate_slice_lhc.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u evaluate.py --dataset lhc --algorithm slice --modellatentdim 20 --observedsamples 1000 --dir /scratch/jb6504/manifold-flow
