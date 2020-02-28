#!/bin/bash

#SBATCH --job-name=t-mfa-p
#SBATCH --output=log_train_mf_power_alternate_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u train.py --modelname small_alternate_wdecay --dataset power --algorithm mf --alternate --samplesize 100000 -i ${SLURM_ARRAY_TASK_ID} --weightdecay 1.e-4 --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname small_alternate_wdecay_shallow_long --dataset power --algorithm mf --alternate --samplesize 100000 --epochs 50 --outerlayers 3 --innerlayers 3 --weightdecay 1.e-4 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname small_alternate_wdecay_long --dataset power --algorithm mf --alternate --samplesize 100000 --epochs 50 --weightdecay 1.e-4 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow

python -u train.py --modelname small_alternate --dataset power --algorithm mf --alternate --samplesize 100000 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname small_alternate_shallow_long --dataset power --algorithm mf --alternate --samplesize 100000 --epochs 50 --outerlayers 3 --innerlayers 3 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname small_alternate_long --dataset power --algorithm mf --alternate --samplesize 100000 --epochs 50 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
