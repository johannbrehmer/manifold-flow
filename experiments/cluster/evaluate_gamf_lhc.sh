#!/bin/bash

#SBATCH --job-name=e-gamf-lhc
#SBATCH --output=log_evaluate_gamf_lhc2_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
cd /scratch/jb6504/manifold-flow/experiments


case ${SLURM_ARRAY_TASK_ID} in
0) python -u evaluate.py --modelname largebs --dataset lhc --algorithm gamf --modellatentdim 9  --observedsamples 100 --dir /scratch/jb6504/manifold-flow ;;
1) python -u evaluate.py --modelname ged_largebs --dataset lhc --algorithm gamf --ged --modellatentdim 9  --observedsamples 100 --dir /scratch/jb6504/manifold-flow ;;
2) python -u evaluate.py --modelname small_largebs --dataset lhc --algorithm gamf --modellatentdim 9  --observedsamples 100 --dir /scratch/jb6504/manifold-flow ;;
3) python -u evaluate.py --modelname small_ged_largebs --dataset lhc --algorithm gamf --ged --modellatentdim 9  --observedsamples 100 --dir /scratch/jb6504/manifold-flow ;;
4) python -u evaluate.py --modelname ged_largebs_long --dataset lhc --algorithm gamf --ged --modellatentdim 9 --observedsamples 100 --dir /scratch/jb6504/manifold-flow ;;
*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac
