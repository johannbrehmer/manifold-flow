#!/bin/bash

sbatch --array 0-59 evaluate_flow_lhc2d_cims.sh
# sbatch --array 60-119 evaluate_flow_lhc2d_cims.sh

sbatch --array 0-59 evaluate_flow_lhc40d_cims.sh
# sbatch --array 60-119 evaluate_flow_lhc40d_cims.sh

sbatch --array 0-59 evaluate_pie_lhc40d_cims.sh
# sbatch --array 60-119 evaluate_pie_lhc40d_cims.sh

sbatch --array 0-35 evaluate_mfs_lhc40d_cims.sh
# sbatch --array 36-47 evaluate_mfs_lhc40d_cims.sh
sbatch --array 48-71 evaluate_mfs_lhc40d_cims.sh
# sbatch --array 72-119 evaluate_mfs_lhc40d_cims.sh

# sbatch --array 0-119 evaluate_mfa_lhc40d_cims.sh

# sbatch --array 0-119 evaluate_emfs_lhc40d_cims.sh

# sbatch --array 0-119 evaluate_emfa_lhc40d_cims.sh
