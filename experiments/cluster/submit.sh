#!/bin/bash


# sbatch --array 1-2 timing.sh

# sbatch  --array 1-2 simulate.sh


sbatch --array 0-2 train_flow_power.sh
sbatch --array 0-2 train_pie_power.sh
sbatch --array 0-2 train_gamf_power.sh
sbatch --array 0-2 train_pie_epsilon_power.sh
sbatch --array 0-80 train_mf_power.sh
sbatch --array 0-9 train_emf_power.sh

#sbatch  --array 1-2 train_flow_lhc2d.sh
#sbatch  --array 1-2 train_flow_lhc.sh
#sbatch  --array 1-2 train_pie_lhc.sh
#sbatch  --array 1-2 train_gamf_lhc.sh
#
#sbatch  --array 1-2 train_flow_spherical.sh
#sbatch  --array 0-0 train_pie_spherical.sh
#sbatch  --array 1-2 train_gamf_spherical.sh
#sbatch  --array 1-2 train_spie_spherical.sh
#sbatch  --array 1-2 train_sgamf_spherical.sh
#sbatch  --array 1-2 train_smf_spherical.sh
#sbatch  --array 1-2 train_pie_epsilon_spherical.sh
#sbatch  --array 21-22 train_mf_spherical.sh
#sbatch  --array 48-49 train_mf_spherical.sh
#sbatch  --array 75-76 train_mf_spherical.sh
#sbatch  --array 3-26 train_emf_spherical.sh
#
#sbatch  --array 1-2 train_flow_csg.sh
#sbatch  --array 0-0 train_pie_csg.sh
#sbatch  --array 1-2 train_gamf_csg.sh
#sbatch  --array 1-2 train_spie_csg.sh
#sbatch  --array 1-2 train_sgamf_csg.sh
#sbatch  --array 1-2 train_smf_csg.sh
#sbatch --array 1-2 train_pie_epsilon_csg.sh
#sbatch  --array 21-22 train_mf_csg.sh
#sbatch  --array 48-49 train_mf_csg.sh
#sbatch  --array 75-76 train_mf_csg.sh
#sbatch --array 3-26 train_emf_csg.sh


# sbatch --array 0-2 evaluate_truth_spherical.sh
# sbatch --array 0-2 evaluate_truth_csg.sh

#sbatch --array 1-2 evaluate_flow_spherical.sh
#sbatch --array 0-0 evaluate_pie_spherical.sh
#sbatch --array 6-8 evaluate_gamf_spherical.sh
#sbatch --array 9-11 evaluate_mf_spherical.sh
#sbatch --array 21-22 evaluate_mf_spherical.sh
#sbatch --array 48-49 evaluate_mf_spherical.sh
#sbatch --array 75-76 evaluate_mf_spherical.sh
#sbatch --array 0-8 evaluate_emf_spherical.sh
#sbatch --array 0-2 evaluate_spie_spherical.sh
#sbatch --array 0-2 evaluate_smf_spherical.sh
#sbatch --array 0-2 evaluate_sgamf_spherical.sh
#sbatch --array 0-2 evaluate_pie_epsilon_spherical.sh

#sbatch --array 0-2 evaluate_flow_csg.sh
#sbatch --array 0-2 evaluate_pie_csg.sh
#sbatch --array 0-35 evaluate_gamf_csg.sh
#sbatch --array 0-80 evaluate_mf_csg.sh
#sbatch --array 0-26 evaluate_emf_csg.sh
#sbatch --array 0-2 evaluate_spie_csg.sh
#sbatch --array 0-2 evaluate_smf_csg.sh
#sbatch --array 0-2 evaluate_sgamf_csg.sh
#sbatch --array 0-2 evaluate_pie_epsilon_csg.sh
#
#sbatch --array 0-2 evaluate_flow_lhc2d.sh
#sbatch --array 0-2 evaluate_flow_lhc.sh
#sbatch --array 0-2 evaluate_pie_lhc.sh
#sbatch --array 0-5 evaluate_gamf_lhc.sh
