#!/bin/bash

#SBATCH --job-name=sim
#SBATCH --output=log_simulate_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

#source activate ml
#export OMP_NUM_THREADS=1
#cd /scratch/jb6504/manifold-flow/experiments

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

python -u generate_data.py --dataset lorenz --train 1000000 --test 10000 --dir $dir
