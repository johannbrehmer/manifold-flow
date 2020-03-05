#!/bin/bash

#SBATCH --job-name=t-gamf-lhc
#SBATCH --output=log_train_gamf_lhc_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u train.py --modelname march --dataset lhc --algorithm gamf --modellatentdim 9 -i ${SLURM_ARRAY_TASK_ID}
python -u train.py --modelname alternate_march --dataset lhc --algorithm gamf --modellatentdim 9 -i ${SLURM_ARRAY_TASK_ID}

#python -u train.py --modelname wdecay_largebs --dataset lhc --algorithm gamf --modellatentdim 9 --genbatchsize 1000 --epochs 100 -i ${SLURM_ARRAY_TASK_ID} --weightdecay 1.e-5 --dir /scratch/jb6504/manifold-flow
#python -u train.py --modelname wdecay_hugebs --dataset lhc --algorithm gamf --modellatentdim 9 --genbatchsize 5000 --epochs 100 -i ${SLURM_ARRAY_TASK_ID} --weightdecay 1.e-5 --dir /scratch/jb6504/manifold-flow
#python -u train.py --modelname wdecay_alternate_largebs --dataset lhc --algorithm gamf --alternate --modellatentdim 9 --genbatchsize 1000 --epochs 100 -i ${SLURM_ARRAY_TASK_ID} --weightdecay 1.e-5 --dir /scratch/jb6504/manifold-flow

#python -u train.py --modelname largebs --dataset lhc --algorithm gamf --modellatentdim 9 --genbatchsize 1000 --epochs 100 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
#python -u train.py --modelname hugebs --dataset lhc --algorithm gamf --modellatentdim 9 --genbatchsize 5000 --epochs 100 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
#python -u train.py --modelname alternate_largebs --dataset lhc --algorithm gamf --alternate --modellatentdim 9 --genbatchsize 1000 --epochs 100 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
