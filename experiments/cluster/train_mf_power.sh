#!/bin/bash

#SBATCH --job-name=t-mf-p
#SBATCH --output=log_train_mf_power_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

run=$((SLURM_ARRAY_TASK_ID / 2))
task=$((SLURM_ARRAY_TASK_ID % 2))
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}, task = ${task}, run = ${run}"

case ${task} in
0) python -u train.py --modelname alternate_march --dataset power --algorithm mf --alternate --splinebins 10 --splinerange 6. --samplesize 100000 -i ${run} ;;
1) python -u train.py --modelname march --dataset power --algorithm mf --splinebins 10 --splinerange 6. --samplesize 100000 -i ${run} ;;
*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac
