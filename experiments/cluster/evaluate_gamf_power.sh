#!/bin/bash

#SBATCH --job-name=e-gamf-p
#SBATCH --output=log_evaluate_gamf_power_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

run=$((SLURM_ARRAY_TASK_ID / 4))
task=$((SLURM_ARRAY_TASK_ID % 4))
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}, task = ${task}, run = ${run}"

case ${task} in
0) python -u evaluate.py --modelname small_largebs --dataset power --algorithm gamf -i ${run} --dir /scratch/jb6504/manifold-flow ;;
1) python -u evaluate.py --modelname small_hugebs --dataset power --algorithm gamf  -i ${run} --dir /scratch/jb6504/manifold-flow ;;
2) python -u evaluate.py --modelname small_largebs_shallow_long --dataset power --algorithm gamf   --outerlayers 3 --innerlayers 3 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
3) python -u evaluate.py --modelname small_largebs_long --dataset power --algorithm gamf   -i ${run} --dir /scratch/jb6504/manifold-flow ;;
*) echo "Nothing to do for job ${task}" ;;
esac
