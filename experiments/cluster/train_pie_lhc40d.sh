#!/bin/bash

#SBATCH --job-name=t-pie-l40
#SBATCH --output=log_train_pie_lhc40d_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u train.py --modelname april --dataset lhc40d --algorithm pie --modellatentdim 14 --splinebins 10 --nllfactor 0.1 --subsets 100 -i ${SLURM_ARRAY_TASK_ID}
# python -u train.py --modelname april_long --load pie_14_lhc40d_april --lr 1.e-5 --dataset lhc40d --algorithm pie --modellatentdim 14 --splinebins 10 --nllfactor 0.1 --subsets 100 -i ${SLURM_ARRAY_TASK_ID}
