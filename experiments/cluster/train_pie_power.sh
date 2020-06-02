#!/bin/bash

#SBATCH --job-name=t-pie-p
#SBATCH --output=log_train_pie_power_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
# #SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/scandal-mf/experiments

# python -u train.py --modelname march --dataset power --algorithm pie --samplesize 100000 --splinebins 10 --splinerange 6. -i ${SLURM_ARRAY_TASK_ID}
python -u train.py -c configs/train_power_march.config --modelname conditionalmanifold_march --algorithm pie --conditionalouter -i ${SLURM_ARRAY_TASK_ID}
