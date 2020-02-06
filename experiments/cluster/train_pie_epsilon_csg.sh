#!/bin/bash

#SBATCH --job-name=t-piee-csg
#SBATCH --output=log_train_pie_epsilon_csg_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

python -u train.py --modelname small_pieepsilon03 --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm pie --pieepsilon 0.3 --samplesize 100000 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname small_pieepsilon01 --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm pie --pieepsilon 0.1 --samplesize 100000 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname small_pieepsilon003 --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm pie --pieepsilon 0.03 --samplesize 100000 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname small_pieepsilon0003 --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm pie --pieepsilon 0.003 --samplesize 100000 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
python -u train.py --modelname small_pieepsilon0001 --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm pie --pieepsilon 0.001 --samplesize 100000 -i ${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/manifold-flow
