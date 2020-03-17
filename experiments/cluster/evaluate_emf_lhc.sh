#!/bin/bash

#SBATCH --job-name=e-emf-l
#SBATCH --output=log_evaluate_emf_lhc_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=3-00:00:00
##SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u evaluate.py --modelname alternate_april --dataset lhc --algorithm emf --modellatentdim 14 --splinebins 10 --observedsamples 50 -i ${SLURM_ARRAY_TASK_ID}
