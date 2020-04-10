#!/bin/bash

#SBATCH --job-name=t-emf-l
#SBATCH --output=log_train_emf_lhc_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

#python -u train.py -c cluster/configs/train_lhc_may.config --modelname sequential_may --algorithm emf --sequential -i ${SLURM_ARRAY_TASK_ID}
#python -u train.py -c cluster/configs/train_lhc_may.config --modelname alternate_may --algorithm emf --alternate -i ${SLURM_ARRAY_TASK_ID}
python -u train.py -c cluster/configs/train_lhc_may.config --modelname sequential_scandal_may --algorithm emf --scandal 5 --sequential -i ${SLURM_ARRAY_TASK_ID}
python -u train.py -c cluster/configs/train_lhc_may.config --modelname alternate_scandal_may --algorithm emf --scandal 5 --alternate -i ${SLURM_ARRAY_TASK_ID}
