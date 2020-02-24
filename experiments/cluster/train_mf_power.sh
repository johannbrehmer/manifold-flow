#!/bin/bash

#SBATCH --job-name=t-mf-p
#SBATCH --output=log_train_mf_power_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
source activate ml
export OMP_NUM_THREADS=1
cd /scratch/jb6504/manifold-flow/experiments

run=$((SLURM_ARRAY_TASK_ID / 9))
task=$((SLURM_ARRAY_TASK_ID % 9))
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}, task = ${task}, run = ${run}"

case ${task} in
0) python -u train.py --modelname small --dataset power --algorithm mf --outercouplingmlp --outercouplinglayers 1 --samplesize 100000 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
1) python -u train.py --modelname small_noprepost --dataset power --algorithm mf --outercouplingmlp --outercouplinglayers 1 --nopretraining --noposttraining --samplesize 100000 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
2) python -u train.py --modelname small_complex --dataset power --algorithm mf --samplesize 100000 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
3) python -u train.py --modelname small_shallow_long --dataset power --algorithm mf --outercouplingmlp --outercouplinglayers 1 --samplesize 100000 --epochs 50 --outerlayers 3 --innerlayers 3 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
4) python -u train.py --modelname small_long --dataset power --algorithm mf --outercouplingmlp --outercouplinglayers 1 --samplesize 100000 --epochs 50 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
5) python -u train.py --modelname small_prepie --dataset power --algorithm mf --outercouplingmlp --outercouplinglayers 1 --prepie --prepostfraction 3 --samplesize 100000 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
6) python -u train.py --modelname small_prepie_long --dataset power --algorithm mf --outercouplingmlp --outercouplinglayers 1 --prepie --prepostfraction 3  --epochs 50 --samplesize 100000 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
7) python -u train.py --modelname small_morenll --dataset power --algorithm mf --outercouplingmlp --outercouplinglayers 1 --samplesize 100000 --addnllfactor 1.0 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
8) python -u train.py --modelname small_morenll_long --dataset power --algorithm mf --outercouplingmlp --outercouplinglayers 1 --samplesize 100000 --addnllfactor 1.0 --epochs 50 -i ${run} --dir /scratch/jb6504/manifold-flow ;;
*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac
