#!/bin/bash

# sbatch --array 0-0 evaluate_flow_lhc2d_cims.sh
# sbatch --array 1-2 evaluate_flow_lhc_cims.sh
# sbatch --array 1-2 evaluate_pie_lhc_cims.sh


sbatch --array 0-2 evaluate_mf_lhc_cims2.sh
sbatch --array 0-2 evaluate_emf_lhc_cims2.sh
sbatch --array 0-5 evaluate_gamf_lhc_cims2.sh

sbatch --array 1-2 evaluate_mf_lhc_cims.sh
sbatch --array 0-1 evaluate_emf_lhc_cims.sh
sbatch --array 0-5 evaluate_gamf_lhc_cims.sh
