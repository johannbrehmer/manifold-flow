#!/usr/bin/env bash

sbatch --array=2-32 train_spherical.sh
sbatch --array=64-64 train_spherical.sh
sbatch --array=128-128 train_spherical.sh
