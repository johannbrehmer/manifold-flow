#!/bin/bash

#SBATCH --job-name=e-p-l
#SBATCH --output=log_evaluate_pae_lhc_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

# module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

run=$((SLURM_ARRAY_TASK_ID / 12))
task=$((SLURM_ARRAY_TASK_ID % 12))
chain=$((task / 3))
true=$((task % 3))
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}, true = ${true}, chain = ${chain}, run = ${run}"

python -u evaluate.py -c configs/evaluate_mf_lhc_june.config --algorithm pae -i $run --trueparam $true --chain $chain --dir /scratch/jb6504/manifold-flow
