#!/bin/bash

#SBATCH --job-name=e-f-l
#SBATCH --output=log_evaluate_flow_lhc_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=1-00:00:00
# #SBATCH --gres=gpu:1

conda activate ml
export PATH="/home/brehmer/miniconda3/envs/ml/bin/:$PATH"
export OMP_NUM_THREADS=1
dir=/data/brehmer/manifold-flow
cd $dir/experiments

run=$((SLURM_ARRAY_TASK_ID / 12))
task=$((SLURM_ARRAY_TASK_ID % 12))
chain=$((task / 3))
true=$((task % 3))
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}, true = ${true}, chain = ${chain}, run = ${run}"

python -u evaluate.py -c configs/evaluate_flow_lhc_june.config -i $run --trueparam $true --chain $chain
