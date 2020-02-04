#!/bin/bash

#SBATCH --job-name=e-emf-csg
#SBATCH --output=log_evaluate_emf_csg_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

case ${SLURM_ARRAY_TASK_ID} in
0) python -u evaluate.py --modelname small --dataset conditional_spherical_gaussian --algorithm emf --outercouplingmlp --outercouplinglayers 1 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow ;;
1) python -u evaluate.py --modelname small --dataset conditional_spherical_gaussian --algorithm emf --outercouplingmlp --outercouplinglayers 1 --epsilon 0.001 --dir /scratch/jb6504/manifold-flow ;;
2) python -u evaluate.py --modelname small --dataset conditional_spherical_gaussian --algorithm emf --outercouplingmlp --outercouplinglayers 1 --epsilon 0.1 --dir /scratch/jb6504/manifold-flow ;;

#6) python -u evaluate.py --modelname small_shallow_long --dataset conditional_spherical_gaussian --algorithm emf --epsilon 0.01 --outerlayers 3 --innerlayers 3 --dir /scratch/jb6504/manifold-flow ;;
#7) python -u evaluate.py --modelname small_shallow_long --dataset conditional_spherical_gaussian --algorithm emf --epsilon 0.001 --outerlayers 3 --innerlayers 3 --dir /scratch/jb6504/manifold-flow ;;
#8) python -u evaluate.py --modelname small_shallow_long --dataset conditional_spherical_gaussian --algorithm emf --epsilon 0.1 --outerlayers 3 --innerlayers 3 --dir /scratch/jb6504/manifold-flow ;;

#9) python -u evaluate.py --modelname small_long --dataset conditional_spherical_gaussian --algorithm emf --epsilon 0.01 --dir /scratch/jb6504/manifold-flow ;;
#10) python -u evaluate.py --modelname small_long --dataset conditional_spherical_gaussian --algorithm emf --epsilon 0.001 --dir /scratch/jb6504/manifold-flow ;;
#11) python -u evaluate.py --modelname small_long --dataset conditional_spherical_gaussian --algorithm emf --epsilon 0.1 --dir /scratch/jb6504/manifold-flow ;;

*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac
