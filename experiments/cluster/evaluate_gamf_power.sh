#!/bin/bash

#SBATCH --job-name=e-gamf-p
#SBATCH --output=log_evaluate_gamf_power_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

run=$((SLURM_ARRAY_TASK_ID / 14))
task=$((SLURM_ARRAY_TASK_ID % 14))
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}, task = ${task}, run = ${run}"

case ${task} in
0) python -u evaluate.py --modelname small_wdecay_largebs --dataset power --algorithm gamf --gridresolution 101 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
1) python -u evaluate.py --modelname small_wdecay_hugebs --dataset power --algorithm gamf  --gridresolution 101 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
2) python -u evaluate.py --modelname small_wdecay_largebs_shallow_long --dataset power --algorithm gamf   --outerlayers 3 --innerlayers 3 --gridresolution 101 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
3) python -u evaluate.py --modelname small_wdecay_largebs_long --dataset power --algorithm gamf   --gridresolution 101 -i ${run} --dir /scratch/jb6504/manifold-flow ;;

4) python -u evaluate.py --modelname small_largebs --dataset power --algorithm gamf --gridresolution 101 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
5) python -u evaluate.py --modelname small_hugebs --dataset power --algorithm gamf  --gridresolution 101 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
6) python -u evaluate.py --modelname small_largebs_shallow_long --dataset power --algorithm gamf   --outerlayers 3 --innerlayers 3 --gridresolution 101 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
7) python -u evaluate.py --modelname small_largebs_long --dataset power --algorithm gamf   --gridresolution 101 -i ${run} --dir /scratch/jb6504/manifold-flow ;;

8) python -u evaluate.py --modelname small_alternate_wdecay --dataset power --algorithm gamf  --gridresolution 101 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
9) python -u evaluate.py --modelname small_alternate_wdecay_shallow_long --dataset power --algorithm gamf   --outerlayers 3 --innerlayers 3 --gridresolution 101 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
10) python -u evaluate.py --modelname small_alternate_wdecay_long --dataset power --algorithm gamf   --gridresolution 101 -i ${run} --dir /scratch/jb6504/manifold-flow ;;

11) python -u evaluate.py --modelname small_alternate --dataset power --algorithm gamf  --gridresolution 101 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
12) python -u evaluate.py --modelname small_alternate_shallow_long --dataset power --algorithm gamf   --outerlayers 3 --innerlayers 3 --gridresolution 101 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
13) python -u evaluate.py --modelname small_alternate_long --dataset power --algorithm gamf   --gridresolution 101 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
*) echo "Nothing to do for job ${task}" ;;
esac
