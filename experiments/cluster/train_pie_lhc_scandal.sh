#!/bin/bash

#SBATCH --job-name=t-ps-l
#SBATCH --output=log_train_pie_lhc_scandal_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

# module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/ANONYMOUS/manifold-flow/experiments

python -u train.py -c configs/train_flow_lhc_june.config --algorithm pie --modelname scandal_june --scandal 2 -i ${SLURM_ARRAY_TASK_ID}
