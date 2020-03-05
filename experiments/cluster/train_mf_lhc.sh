#!/bin/bash

#SBATCH --job-name=t-mf-l
#SBATCH --output=log_train_mf_lhc2_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u train.py --modelname alternate_march --dataset lhc --algorithm mf --alternate --modellatentdim 9 -i ${SLURM_ARRAY_TASK_ID}

# python -u train.py --modelname alternate --dataset lhc --algorithm mf --alternate --modellatentdim 9 --epochs 100 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
# python -u train.py --modelname alternate_wdecay --dataset lhc --algorithm mf --alternate --modellatentdim 9 --weightdecay 1.e-5 --epochs 100 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
