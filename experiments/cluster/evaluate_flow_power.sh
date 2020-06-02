#!/bin/bash

#SBATCH --job-name=e-sf-p
#SBATCH --output=log_evaluate_flow_power_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/ANONYMOUS/manifold-flow/experiments

python -u evaluate.py --modelname march --dataset power --algorithm flow --splinebins 10 --splinerange 6.  --gridresolution 101 -i ${SLURM_ARRAY_TASK_ID}
