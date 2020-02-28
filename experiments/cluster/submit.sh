#!/bin/bash


#sbatch  --array 1-2 simulate.sh


#sbatch  --array 0-2 train_flow_lhc2d.sh
#sbatch  --array 0-2 train_flow_lhc.sh
#sbatch  --array 0-2 train_pie_lhc.sh
#sbatch  --array 0-2 train_gamf_lhc.sh
#sbatch --array 0-2 train_mf_lhc.sh
#sbatch --array 0-2 train_emf_lhc.sh

#sbatch --array 0-2 train_flow_power.sh
#sbatch --array 0-2 train_pie_power.sh
#sbatch --array 0-2 train_gamf_power.sh
#sbatch --array 0-2 train_pie_epsilon_power.sh
#sbatch --array 0-2 train_mf_alternating_power.sh
#sbatch --array 0-2 train_emf_alternating_power.sh
#sbatch --array 0-12 train_mf_power.sh
#sbatch --array 0-12 train_emf_power.sh


#sbatch --array 0-2 evaluate_flow_lhc2d.sh
#sbatch --array 0-2 evaluate_flow_lhc.sh
#sbatch --array 0-2 evaluate_pie_lhc.sh
#sbatch --array 0-2 evaluate_mf_lhc.sh  <--
#sbatch --array 0-2 evaluate_emf_lhc.sh  <--
#sbatch --array 0-9 evaluate_gamf_lhc.sh

#sbatch --array 0-2 evaluate_truth_power.sh
sbatch --array 0-2 evaluate_flow_power.sh
sbatch --array 0-2 evaluate_pie_power.sh
sbatch --array 0-2 evaluate_gamf_power.sh
sbatch --array 0-2 evaluate_pie_epsilon_power.sh
sbatch --array 0-20 evaluate_mf_power.sh  <--
sbatch --array 0-20 evaluate_emf_power.sh  <--
