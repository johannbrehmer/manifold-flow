#!/bin/bash

#SBATCH --job-name=p-mf-lhc
#SBATCH --output=log_paramscan_mf_lhc_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=2-00:00:00
# #SBATCH --gres=gpu:1

conda activate ml
export PATH="/home/brehmer/miniconda3/envs/ml/bin/:$PATH"
export OMP_NUM_THREADS=1
dir=/data/brehmer/manifold-flow
cd $dir/experiments

python -u paramscan.py -c configs/paramscan_lhc_june.config --paramscanstudyname paramscan_june_${SLURM_ARRAY_TASK_ID} --dir $dir
