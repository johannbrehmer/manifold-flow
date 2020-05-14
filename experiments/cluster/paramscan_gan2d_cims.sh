#!/bin/bash

#SBATCH --job-name=p-g2-lhc
#SBATCH --output=log_paramscan_mf_gan2d_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_12gb

module load cuda-10.2
conda activate ml
export PATH="/home/brehmer/miniconda3/envs/ml/bin/:$PATH"
export OMP_NUM_THREADS=1
dir=/data/brehmer/manifold-flow
cd $dir/experiments

python -u paramscan_images.py -c configs/paramscan_gan2d_may.config --paramscanstudyname paramscan_gan2d_may_${SLURM_ARRAY_TASK_ID}
