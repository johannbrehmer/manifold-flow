#!/usr/bin/env bash

conda activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments

#python -u train.py -c cluster/configs/train_mfmf_lorenz_april.config --modelname april2
python -u train.py -c cluster/configs/train_mfmf_lorenz_april.config --modelname april2 --load checkpoints/mf_2_lorenz_april2_epoch_A45
