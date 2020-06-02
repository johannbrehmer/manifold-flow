#!/bin/bash

#SBATCH --job-name=t-mf-p
#SBATCH --output=log_train_mf_power_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/ANONYMOUS/manifold-flow/experiments

# python -u train.py --modelname alternate_march --dataset power --algorithm mf --alternate --splinebins 10 --splinerange 6. --samplesize 100000 -i ${SLURM_ARRAY_TASK_ID}
# python -u train.py --modelname march --dataset power --algorithm mf --splinebins 10 --splinerange 6. --samplesize 100000 -i ${SLURM_ARRAY_TASK_ID}
python -u train.py --modelname sequential_march --dataset power --algorithm mf --sequential --splinebins 10 --splinerange 6. --samplesize 100000 -i ${SLURM_ARRAY_TASK_ID}
