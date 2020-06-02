#!/bin/bash

################################################################################
# LHC eval jobs
################################################################################


sbatch --array 0-119 evaluate_mf_lhc_cims.sh
sbatch --array 0-119 evaluate_mf_scandal_lhc_cims.sh
sbatch --array 0-119 evaluate_emf_lhc_cims.sh
sbatch --array 0-119 evaluate_emf_scandal_lhc_cims.sh
sbatch --array 0-119 evaluate_flow_lhc_cims.sh
sbatch --array 0-119 evaluate_flow_scandal_lhc_cims.sh
sbatch --array 0-119 evaluate_pie_lhc_cims.sh
sbatch --array 0-119 evaluate_pie_scandal_lhc_cims.sh
sbatch --array 0-119 evaluate_pie_cm_lhc_cims.sh
sbatch --array 0-119 evaluate_pie_cm_scandal_lhc_cims.sh



################################################################################
# GAN2D train jobs
################################################################################

#sbatch --array 0-9 train_mf_gan2d_cims.sh
#sbatch --array 0-9 train_emf_gan2d_cims.sh
#sbatch --array 0-9 train_pie_gan2d_cims.sh
#sbatch --array 0-9 train_flow_gan2d_cims.sh
