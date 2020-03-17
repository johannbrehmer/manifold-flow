#!/bin/bash

#SBATCH --job-name=e-sf-l2d
#SBATCH --output=log_evaluate_flow_lhc2d_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=3-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u evaluate.py --modelname april --dataset lhc2d --algorithm flow --modellatentdim 2 --splinebins 10 --observedsamples 50 -i ${SLURM_ARRAY_TASK_ID}
