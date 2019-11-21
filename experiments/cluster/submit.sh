#!/usr/bin/env bash

# sbatch simulate.sh

sbatch train_flow_spherical.sh
sbatch train_pie_spherical.sh
sbatch --array=0-8 train_mf_spherical.sh
