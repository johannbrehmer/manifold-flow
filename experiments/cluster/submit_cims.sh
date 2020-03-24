#!/bin/bash

# Run 1
sbatch --array 0-4 evaluate_missing_lhc_cims.sh

# Run 1
# sbatch --array 0-11 evaluate_flow_lhc2d_cims.sh  # done
# sbatch --array 0-11 evaluate_flow_lhc_cims.sh  # done
# sbatch --array 0-11 evaluate_pie_lhc_cims.sh  # done
# sbatch --array 0-11 evaluate_mf_lhc_cims.sh  # submitted
# sbatch --array 0-11 evaluate_emf_lhc_cims.sh  # submitted
#sbatch --array 0-11 evaluate_gamf_lhc_cims.sh
#sbatch --array 0-11 evaluate_gamf2_lhc_cims.sh

# sleep 60

# Run 2-3
# sbatch --array 12-35 evaluate_flow_lhc2d_cims.sh  # done
# sbatch --array 12-35 evaluate_flow_lhc_cims.sh  # done
# sbatch --array 12-35 evaluate_pie_lhc_cims.sh  # submitted
# sbatch --array 12-35 evaluate_mf_lhc_cims.sh  # submitted
# sbatch --array 12-23 evaluate_emf_lhc_cims.sh  # submitted
# sbatch --array 24-35 evaluate_emf_lhc_cims.sh  # submitted
# sbatch --array 12-35 evaluate_gamf_lhc_cims.sh
# sbatch --array 12-35 evaluate_gamf2_lhc_cims.sh

# sleep 60

# Run 4-5
# sbatch --array 36-59 evaluate_flow_lhc2d_cims.sh  # submitted
# sbatch --array 36-59 evaluate_flow_lhc_cims.sh  # submitted
# sbatch --array 36-59 evaluate_pie_lhc_cims.sh  # submitted
sbatch --array 36-59 evaluate_mf_lhc_cims.sh
# sbatch --array 36-59 evaluate_emf_lhc_cims.sh
# sbatch --array 36-59 evaluate_gamf_lhc_cims.sh
# sbatch --array 36-59 evaluate_gamf2_lhc_cims.sh
