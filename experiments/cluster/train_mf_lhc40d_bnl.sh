#!/bin/bash

#SBATCH --job-name=t-mf-l
#SBATCH --output=log_train_mf_lhc_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /sdcc/u/brehmer/manifold-flow/experiments

python -u train.py -c cluster/configs/train_mfmf_lhc_june.config --modelname june -i ${SLURM_ARRAY_TASK_ID}
# python -u train.py -c cluster/configs/train_mfmf_lhc_june.config --modelname scandal_june --scandal 1 -i ${SLURM_ARRAY_TASK_ID}
