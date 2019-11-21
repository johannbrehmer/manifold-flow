#!/bin/bash

#SBATCH --job-name=mf-sim
#SBATCH --output=log_simulate.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate madminer
cd /scratch/jb6504/manifold-flow/experiments

python -u generate_data/spherical_simulator.py --datadim 15 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow

python -u generate_data/spherical_simulator.py --datadim 15 --epsilon 0.001 --dir /scratch/jb6504/manifold-flow
python -u generate_data/spherical_simulator.py --datadim 15 --epsilon 0.003 --dir /scratch/jb6504/manifold-flow
python -u generate_data/spherical_simulator.py --datadim 15 --epsilon 0.03 --dir /scratch/jb6504/manifold-flow
python -u generate_data/spherical_simulator.py --datadim 15 --epsilon 0.1 --dir /scratch/jb6504/manifold-flow

python -u generate_data/spherical_simulator.py --datadim 11 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
python -u generate_data/spherical_simulator.py --datadim 13 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
python -u generate_data/spherical_simulator.py --datadim 20 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
python -u generate_data/spherical_simulator.py --datadim 30 --epsilon 0.01 --dir /scratch/jb6504/manifold-flow
