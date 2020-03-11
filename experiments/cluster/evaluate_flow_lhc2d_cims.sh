#!/bin/bash

#SBATCH --job-name=e-sf-l2d
#SBATCH --output=log_evaluate_flow_lhc2d_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=2-00:00:00
# #SBATCH --gres=gpu:1

source /home/brehmer/miniconda3/etc/profile.d/conda.sh
conda activate ml
export OMP_NUM_THREADS=1
dir=/home/brehmer/manifold-flow

cd $dir/experiments
python -u evaluate.py --modelname march --dataset lhc2d --algorithm flow --modellatentdim 2 --observedsamples 100 -i ${SLURM_ARRAY_TASK_ID} --dir $dir
