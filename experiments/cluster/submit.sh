#!/usr/bin/env bash

# sbatch simulate.sh

sbatch timing.sh

# sbatch train_flow_spherical.sh
# sbatch train_pie_spherical.sh
sbatch --array=0-4 train_mf_spherical.sh
# sbatch train_flow_csg.sh
# sbatch train_pie_csg.sh
sbatch --array=0-4 train_mf_csg.sh
# sbatch train_opie_csg.sh
sbatch --array=0-4 train_omf_csg.sh

#sbatch evaluate_flow_spherical.sh
#sbatch evaluate_pie_spherical.sh
#sbatch evaluate_mf_spherical.sh
