#!/bin/bash

#SBATCH --job-name=aef-t-cifar
#SBATCH --output=log_train_cifar_flow.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate madminer
cd /scratch/jb6504/autoencoded-flow2/

python -u train.py cifar --dataset cifar --dir /scratch/jb6504/autoencoded-flow2 --lr 5.e-4 --outer 3 --debug
