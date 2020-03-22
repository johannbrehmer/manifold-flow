#!/bin/bash

#SBATCH --job-name=t-mf-l
#SBATCH --output=log_train_mf_lhc_%a_long.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

# python -u train.py --modelname alternate_april --dataset lhc --algorithm mf --alternate --modellatentdim 14 --splinebins 10 --nllfactor 0.1 --subsets 100 -i ${SLURM_ARRAY_TASK_ID}
python -u train.py --modelname alternate_april_long --load mf_14_lhc_alternate_april --lr 1.e-4 --dataset lhc --algorithm mf --alternate --modellatentdim 14 --splinebins 10 --nllfactor 0.1 --subsets 100 -i ${SLURM_ARRAY_TASK_ID}
