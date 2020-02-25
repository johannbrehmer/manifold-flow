#!/bin/bash

#SBATCH --job-name=e-piee-p
#SBATCH --output=log_evaluate_pie_epsilon_power_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u evaluate.py --modelname small_pieepsilon01 --dataset power --algorithm pie --pieepsilon 0.1 --gridresolution 101 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_pieepsilon003 --dataset power --algorithm pie --pieepsilon 0.03 --gridresolution 101 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_pieepsilon0003 --dataset power --algorithm pie --pieepsilon 0.003 --gridresolution 101 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_pieepsilon0001 --dataset power --algorithm pie --pieepsilon 0.001 --gridresolution 101 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_pieepsilon00003 --dataset power --algorithm pie --pieepsilon 0.0003 --gridresolution 101 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u evaluate.py --modelname small_pieepsilon00001 --dataset power --algorithm pie --pieepsilon 0.0001 --gridresolution 101 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
