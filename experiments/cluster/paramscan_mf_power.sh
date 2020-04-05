#!/bin/bash

#SBATCH --job-name=p-mf-p
#SBATCH --output=log_paramscan_mf_power_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
##SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u paramscan.py --paramscanstudyname mfpower${SLURM_ARRAY_TASK_ID}
