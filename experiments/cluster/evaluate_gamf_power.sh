#!/bin/bash

#SBATCH --job-name=e-gamf-p
#SBATCH --output=log_evaluate_gamf_power_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=2-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/ANONYMOUS/manifold-flow/experiments

run=$((SLURM_ARRAY_TASK_ID / 2))
task=$((SLURM_ARRAY_TASK_ID % 2))
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}, task = ${task}, run = ${run}"

case ${task} in
0) python -u evaluate.py --modelname march --dataset power --algorithm gamf --splinebins 10 --splinerange 6. --gridresolution 101 -i ${run} ;;
1) python -u evaluate.py --modelname alternate_march --dataset power --algorithm gamf --splinebins 10 --splinerange 6. --gridresolution 101 -i ${run} ;;
*) echo "Nothing to do for job ${task}" ;;
esac
