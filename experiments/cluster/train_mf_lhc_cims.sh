#!/bin/bash

#SBATCH --job-name=t-mf-l
#SBATCH --output=log_train_mf_lhc_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

conda activate ml
export PATH="/home/brehmer/miniconda3/envs/ml/bin/:$PATH"
export OMP_NUM_THREADS=1
dir=/home/brehmer/manifold-flow
cd $dir/experiments

python -u train.py --modelname alternate_april --dataset lhc --algorithm mf --alternate --modellatentdim 14 --splinebins 10 --nllfactor 0.1 --subsets 100 -i ${SLURM_ARRAY_TASK_ID}
