#!/usr/bin/env bash

# sbatch simulate.sh

# sbatch timing.sh

# sbatch train_flow_spherical.sh
# sbatch train_pie_spherical.sh
# sbatch train_slice_spherical.sh
# sbatch --array=0-2 train_mf_spherical.sh
sbatch train_gamf_spherical.sh
sbatch train_hybrid_spherical.sh

# sbatch train_flow_csg.sh
# sbatch train_pie_csg.sh
# sbatch train_slice_csg.sh
# sbatch --array=0-2 train_mf_csg.sh
sbatch train_gamf_csg.sh
sbatch train_hybrid_csg.sh

# sbatch evaluate_flow_spherical.sh
# sbatch evaluate_pie_spherical.sh
# sbatch evaluate_slice_spherical.sh
# sbatch evaluate_mf_spherical.sh

# sbatch evaluate_flow_csg.sh
# sbatch evaluate_pie_csg.sh
# sbatch evaluate_slice_csg.sh
# sbatch evaluate_mf_csg.sh
