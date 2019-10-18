#!/bin/bash

#SBATCH --job-name=mwe
#SBATCH --output=log_mwe.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate ml
cd /scratch/jb6504/manifold_flow_mwe

python -u mwe_pytorch.py
