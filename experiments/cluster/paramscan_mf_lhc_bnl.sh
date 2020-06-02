#!/bin/bash

#SBATCH -p usatlas
#SBATCH -t 1-00:00:00
#SBATCH --qos=usatlas
#SBATCH --account=tier3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --job-name=p-mf-lhc
#SBATCH --output=log_paramscan_mf_lhc_%a.log

source ~/.bashrc
module load cuda/9.0
conda activate ml
export OMP_NUM_THREADS=1
cd /sdcc/u/brehmer/manifold-flow/experiments

python -u paramscan.py -c configs/paramscan_lhc_june.config --paramscanstudyname paramscan_june_${SLURM_ARRAY_TASK_ID}
