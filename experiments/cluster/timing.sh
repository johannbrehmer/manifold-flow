#!/bin/bash

#SBATCH --job-name=mf-t-sf-csg
#SBATCH --output=log_train_flow_csg.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate ml
cd /scratch/jb6504/manifold-flow/experiments

python -u timing.py --algorithm flow --dir /scratch/jb6504/manifold-flow
python -u timing.py --algorithm pie --dir /scratch/jb6504/manifold-flow
python -u timing.py --algorithm mf --dir /scratch/jb6504/manifold-flow
python -u timing.py --algorithm mf --outercouplinglayers 1 --hidden 100 --dir /scratch/jb6504/manifold-flow
python -u timing.py --algorithm mf --outercouplingmlp --dir /scratch/jb6504/manifold-flow
python -u timing.py --algorithm mf --outercouplingmlp --outercouplinglayers 1 --hidden 100 --dir /scratch/jb6504/manifold-flow
