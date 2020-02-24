#!/bin/bash

#SBATCH --job-name=e-pie-p
#SBATCH --output=log_evaluate_pie_power_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u evaluate.py --modelname small_shallow_long --dataset power --algorithm pie  --outerlayers 3 --innerlayers 3 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_long --dataset power --algorithm pie  -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small --dataset power --algorithm pie -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
