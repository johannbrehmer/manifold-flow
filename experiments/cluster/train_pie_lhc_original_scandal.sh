#!/bin/bash

#SBATCH --job-name=t-osp-l
#SBATCH --output=log_train_pie_lhc_original_scandal_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml2
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow2/experiments

python -u train.py -c configs/train_flow_lhc_june.config --algorithm pie --conditionalouter --scandal 2.0 --modelname conditionalmanifold_scandal_june --resume 40 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow2
