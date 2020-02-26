#!/bin/bash

#SBATCH --job-name=e-mf-sg
#SBATCH --output=log_evaluate_mf_spherical_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

run=$((SLURM_ARRAY_TASK_ID / 27))
task=$((SLURM_ARRAY_TASK_ID % 27))
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}, task = ${task}, run = ${run}"

case ${task} in
0) python -u evaluate.py --modelname small --dataset spherical_gaussian --algorithm mf --outercouplingmlp --outercouplinglayers 1 --epsilon 0.01 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
1) python -u evaluate.py --modelname small --dataset spherical_gaussian --algorithm mf --outercouplingmlp --outercouplinglayers 1 --epsilon 0.001 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
2) python -u evaluate.py --modelname small --dataset spherical_gaussian --algorithm mf --outercouplingmlp --outercouplinglayers 1 --epsilon 0.1 -i ${run} --dir /scratch/jb6504/manifold-flow ;;

3) python -u evaluate.py --modelname small_noprepost --dataset spherical_gaussian --epsilon 0.01 --algorithm mf --outercouplingmlp --outercouplinglayers 1 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
4) python -u evaluate.py --modelname small_noprepost --dataset spherical_gaussian --epsilon 0.001 --algorithm mf --outercouplingmlp --outercouplinglayers 1 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
5) python -u evaluate.py --modelname small_noprepost --dataset spherical_gaussian --epsilon 0.1  --algorithm mf --outercouplingmlp --outercouplinglayers 1 -i ${run} --dir /scratch/jb6504/manifold-flow ;;

6) python -u evaluate.py --modelname small_complex --dataset spherical_gaussian --epsilon 0.01 --algorithm mf -i ${run} --dir /scratch/jb6504/manifold-flow ;;
7) python -u evaluate.py --modelname small_complex --dataset spherical_gaussian --epsilon 0.001 --algorithm mf -i ${run} --dir /scratch/jb6504/manifold-flow ;;
8) python -u evaluate.py --modelname small_complex --dataset spherical_gaussian --epsilon 0.1  --algorithm mf -i ${run} --dir /scratch/jb6504/manifold-flow ;;

9) python -u evaluate.py --modelname small_shallow_long --dataset spherical_gaussian --algorithm mf --epsilon 0.01 --outercouplingmlp --outercouplinglayers 1 --outerlayers 3 --innerlayers 3 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
10) python -u evaluate.py --modelname small_shallow_long --dataset spherical_gaussian --algorithm mf --epsilon 0.001 --outercouplingmlp --outercouplinglayers 1 --outerlayers 3 --innerlayers 3 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
11) python -u evaluate.py --modelname small_shallow_long --dataset spherical_gaussian --algorithm mf --epsilon 0.1 --outercouplingmlp --outercouplinglayers 1 --outerlayers 3 --innerlayers 3 -i ${run} --dir /scratch/jb6504/manifold-flow ;;

12) python -u evaluate.py --modelname small_long --dataset spherical_gaussian --algorithm mf --outercouplingmlp --outercouplinglayers 1 --epsilon 0.01 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
13) python -u evaluate.py --modelname small_long --dataset spherical_gaussian --algorithm mf --outercouplingmlp --outercouplinglayers 1 --epsilon 0.001 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
14) python -u evaluate.py --modelname small_long --dataset spherical_gaussian --algorithm mf --outercouplingmlp --outercouplinglayers 1 --epsilon 0.1 -i ${run} --dir /scratch/jb6504/manifold-flow ;;

15) python -u evaluate.py --modelname small_prepie --dataset spherical_gaussian --epsilon 0.01 --algorithm mf --outercouplingmlp --outercouplinglayers 1 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
16) python -u evaluate.py --modelname small_prepie --dataset spherical_gaussian --epsilon 0.001 --algorithm mf --outercouplingmlp --outercouplinglayers 1 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
17) python -u evaluate.py --modelname small_prepie --dataset spherical_gaussian --epsilon 0.1  --algorithm mf --outercouplingmlp --outercouplinglayers 1 -i ${run} --dir /scratch/jb6504/manifold-flow ;;

18) python -u evaluate.py --modelname small_prepie_long --dataset spherical_gaussian --epsilon 0.01 --algorithm mf --outercouplingmlp --outercouplinglayers 1 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
19) python -u evaluate.py --modelname small_prepie_long --dataset spherical_gaussian --epsilon 0.001 --algorithm mf --outercouplingmlp --outercouplinglayers 1 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
20) python -u evaluate.py --modelname small_prepie_long --dataset spherical_gaussian --epsilon 0.1  --algorithm mf --outercouplingmlp --outercouplinglayers 1 -i ${run} --dir /scratch/jb6504/manifold-flow ;;

21) python -u evaluate.py --modelname small_morenll --dataset spherical_gaussian --algorithm mf --outercouplingmlp --outercouplinglayers 1 --epsilon 0.01 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
22) python -u evaluate.py --modelname small_morenll --dataset spherical_gaussian --algorithm mf --outercouplingmlp --outercouplinglayers 1 --epsilon 0.001 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
23) python -u evaluate.py --modelname small_morenll --dataset spherical_gaussian --algorithm mf --outercouplingmlp --outercouplinglayers 1 --epsilon 0.1 -i ${run} --dir /scratch/jb6504/manifold-flow ;;

24) python -u evaluate.py --modelname small_morenll_long --dataset spherical_gaussian --algorithm mf --outercouplingmlp --outercouplinglayers 1 --epsilon 0.01 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
25) python -u evaluate.py --modelname small_morenll_long --dataset spherical_gaussian --algorithm mf --outercouplingmlp --outercouplinglayers 1 --epsilon 0.001 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
26) python -u evaluate.py --modelname small_morenll_long --dataset spherical_gaussian --algorithm mf --outercouplingmlp --outercouplinglayers 1 --epsilon 0.1 -i ${run} --dir /scratch/jb6504/manifold-flow ;;

*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac
