#!/bin/bash

#SBATCH --job-name=e-mf-p
#SBATCH --output=log_evaluate_mf_power_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=2-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

run=$((SLURM_ARRAY_TASK_ID / 2))
task=$((SLURM_ARRAY_TASK_ID % 2))
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}, task = ${task}, run = ${run}"

case ${task} in
0) python -u evaluate.py --modelname alternate_march --dataset power --algorithm mf --splinebins 10 --splinerange 6. --gridresolution 101 -i ${run} ;;
1) python -u evaluate.py --modelname march --dataset power --algorithm mf --splinebins 10 --splinerange 6. --gridresolution 101 -i ${run} ;;
*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac
