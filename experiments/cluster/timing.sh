#!/bin/bash

#SBATCH --job-name=mf-time
#SBATCH --output=log_timing.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u timing.py --algorithm flow --dir /scratch/jb6504/manifold-flow
python -u timing.py --algorithm pie --dir /scratch/jb6504/manifold-flow
python -u timing.py --algorithm mf --dir /scratch/jb6504/manifold-flow
python -u timing.py --algorithm mf --outercouplinglayers 1 --outercouplinghidden 100 --dir /scratch/jb6504/manifold-flow
python -u timing.py --algorithm mf --outercouplingmlp --dir /scratch/jb6504/manifold-flow
python -u timing.py --algorithm mf --outercouplingmlp --outercouplinglayers 1 --outercouplinghidden 100 --dir /scratch/jb6504/manifold-flow
