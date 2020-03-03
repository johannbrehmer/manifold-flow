#!/bin/bash

#SBATCH --job-name=e-gamf-lhc
#SBATCH --output=log_evaluate_gamf_lhc2_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

run=$((task / 6))
task=$((task % 6))
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}, task = ${task}, run = ${run}"

case ${task} in
0) python -u evaluate.py --modelname wdecay_largebs --dataset lhc --algorithm gamf --modellatentdim 9  --observedsamples 100 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
1) python -u evaluate.py --modelname wdecay_hugebs --dataset lhc --algorithm gamf --modellatentdim 9  --observedsamples 100 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
2) python -u evaluate.py --modelname largebs --dataset lhc --algorithm gamf --modellatentdim 9  --observedsamples 100 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
3) python -u evaluate.py --modelname hugebs --dataset lhc --algorithm gamf --modellatentdim 9  --observedsamples 100 -i ${run} --dir /scratch/jb6504/manifold-flow ;;

4) python -u evaluate.py --modelname wdecay_alternate_largebs --dataset lhc --algorithm gamf --modellatentdim 9  --observedsamples 100 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
5) python -u evaluate.py --modelname alternate_largebs --dataset lhc --algorithm gamf --modellatentdim 9  --observedsamples 100 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
*) echo "Nothing to do for job ${task}" ;;
esac
