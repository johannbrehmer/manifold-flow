#!/bin/bash

#SBATCH --job-name=e-miss
#SBATCH --output=log_evaluate_missing_lhc_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=2-00:00:00
# #SBATCH --gres=gpu:1

conda activate ml
export PATH="/home/brehmer/miniconda3/envs/ml/bin/:$PATH"
export OMP_NUM_THREADS=1
dir=/home/brehmer/manifold-flow
cd $dir/experiments

python -u evaluate.py --modelname alternate_april --dataset lhc --algorithm emf --modellatentdim 14 --splinebins 10 -i $SLURM_ARRAY_TASK_ID --skipinference --dir $dir
python -u evaluate.py --modelname alternate_april --dataset lhc --algorithm mf --modellatentdim 14 --splinebins 10 -i $SLURM_ARRAY_TASK_ID --skipinference --dir $dir
python -u evaluate.py --modelname april --dataset lhc --algorithm flow --modellatentdim 14 --splinebins 10 -i $SLURM_ARRAY_TASK_ID --skipinference --dir $dir
python -u evaluate.py --modelname april --dataset lhc --algorithm pie --modellatentdim 14 --splinebins 10 -i $SLURM_ARRAY_TASK_ID --skipinference --dir $dir
