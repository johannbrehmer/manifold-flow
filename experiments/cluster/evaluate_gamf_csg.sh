#!/bin/bash

#SBATCH --job-name=e-gamf-csg
#SBATCH --output=log_evaluate_gamf_csg_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
cd /scratch/jb6504/manifold-flow/experiments


case ${SLURM_ARRAY_TASK_ID} in
0) python -u evaluate.py --modelname small_largebs --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm gamf --dir /scratch/jb6504/manifold-flow ;;
1) python -u evaluate.py --modelname small_largebs --dataset conditional_spherical_gaussian --epsilon 0.001 --algorithm gamf --dir /scratch/jb6504/manifold-flow ;;
2) python -u evaluate.py --modelname small_largebs --dataset conditional_spherical_gaussian --epsilon 0.1 --algorithm gamf --dir /scratch/jb6504/manifold-flow ;;
3) python -u evaluate.py --modelname small_ged_largebs --dataset conditional_spherical_gaussian --epsilon 0.01 --algorithm gamf --ged --dir /scratch/jb6504/manifold-flow ;;
4) python -u evaluate.py --modelname small_ged_largebs --dataset conditional_spherical_gaussian --epsilon 0.001 --algorithm gamf --ged --dir /scratch/jb6504/manifold-flow ;;
5) python -u evaluate.py --modelname small_ged_largebs --dataset conditional_spherical_gaussian --epsilon 0.1 --algorithm gamf --ged --dir /scratch/jb6504/manifold-flow ;;
*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac


case ${SLURM_ARRAY_TASK_ID} in
0) python -u evaluate.py --modelname largebs --dataset conditional_spherical_gaussian --algorithm gamf --epsilon 0.01 --dir /scratch/jb6504/manifold-flow ;;
1) python -u evaluate.py --modelname largebs --dataset conditional_spherical_gaussian --algorithm gamf --epsilon 0.001 --dir /scratch/jb6504/manifold-flow ;;
2) python -u evaluate.py --modelname largebs --dataset conditional_spherical_gaussian --algorithm gamf --epsilon 0.1 --dir /scratch/jb6504/manifold-flow ;;
*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac
