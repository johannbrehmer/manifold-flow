#!/bin/bash


# sbatch --array 1-2 timing.sh

# sbatch  --array 1-2 simulate.sh


#sbatch  --array 1-2 train_flow_lhc2d.sh
#sbatch  --array 1-2 train_flow_lhc.sh
#sbatch  --array 1-2 train_pie_lhc.sh
#sbatch  --array 1-2 train_gamf_lhc.sh
#
#sbatch  --array 1-2 train_flow_spherical.sh
#sbatch  --array 1-2 train_pie_spherical.sh
#sbatch  --array 1-2 train_gamf_spherical.sh
#sbatch  --array 1-2 train_spie_spherical.sh
#sbatch  --array 1-2 train_sgamf_spherical.sh
#sbatch  --array 1-2 train_smf_spherical.sh
#sbatch  --array 1-2 train_pie_epsilon_spherical.sh
#sbatch  --array 27-80 train_mf_spherical.sh
#sbatch  --array 3-26 train_emf_spherical.sh
#
#sbatch  --array 1-2 train_flow_csg.sh
#sbatch  --array 1-2 train_pie_csg.sh
#sbatch  --array 1-2 train_gamf_csg.sh
#sbatch  --array 1-2 train_spie_csg.sh
#sbatch  --array 1-2 train_sgamf_csg.sh
#sbatch  --array 1-2 train_smf_csg.sh
#sbatch --array 1-2 train_pie_epsilon_csg.sh
#sbatch  --array 27-80 train_mf_csg.sh
#sbatch --array 3-26 train_emf_csg.sh

# sbatch --array 0-2 evaluate_truth_spherical.sh
# sbatch --array 0-2 evaluate_truth_csg.sh


#sbatch --array 1-2 evaluate_flow_spherical.sh
#sbatch --array 1-2 evaluate_pie_spherical.sh
#sbatch --array 12-35 evaluate_gamf_spherical.sh
#sbatch --array 27-80 evaluate_mf_spherical.sh
#sbatch --array 3-8 evaluate_emf_spherical.sh
#sbatch --array 1-2 evaluate_spie_spherical.sh
#sbatch --array 1-2 evaluate_smf_spherical.sh
#sbatch --array 1-2 evaluate_sgamf_spherical.sh
#sbatch --array 1-2 evaluate_pie_epsilon_spherical.sh

#sbatch --array 1-2 evaluate_flow_csg.sh
#sbatch --array 1-2 evaluate_pie_csg.sh
#sbatch --array 12-35 evaluate_gamf_csg.sh
#sbatch --array 27-80 evaluate_mf_csg.sh
#sbatch --array 3-8 evaluate_emf_csg.sh
#sbatch --array 1-2 evaluate_spie_csg.sh
#sbatch --array 1-2 evaluate_smf_csg.sh
#sbatch --array 1-2 evaluate_sgamf_csg.sh
#sbatch --array 1-2 evaluate_pie_epsilon_csg.sh


sbatch --array 0-2 evaluate_flow_lhc2d.sh
sbatch --array 0-2 evaluate_flow_lhc.sh
sbatch --array 0-2 evaluate_pie_lhc.sh
sbatch --array 0-5 evaluate_gamf_lhc.sh
