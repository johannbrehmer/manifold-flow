#!/bin/bash

# Last update master: April 13, 10:00
# Last update scandal: --



# sbatch --array 0-107 evaluate_flow_lhc2d_cims.sh  # done
sbatch --array 108-119 evaluate_flow_lhc2d_cims.sh

# sbatch --array 0-107 evaluate_flow_lhc40d_cims.sh  # done
sbatch --array 108-119 evaluate_flow_lhc40d_cims.sh

# sbatch --array 0-107 evaluate_pie_lhc40d_cims.sh  # done
sbatch --array 108-119 evaluate_pie_lhc40d_cims.sh

# sbatch --array 0-35 evaluate_mfs_lhc40d_cims.sh  # done
sbatch --array 36-47 evaluate_mfs_lhc40d_cims.sh
# sbatch --array 48-107 evaluate_mfs_lhc40d_cims.sh  # done
sbatch --array 108-119 evaluate_mfs_lhc40d_cims.sh

# sbatch --array 0-35 evaluate_mfa_lhc40d_cims.sh  # done
# sbatch --array 36-47 evaluate_mfa_lhc40d_cims.sh
# sbatch --array 48-83 evaluate_mfa_lhc40d_cims.sh  # done
sbatch --array 84-107 evaluate_mfa_lhc40d_cims.sh
# sbatch --array 108-119 evaluate_mfa_lhc40d_cims.sh

# sbatch --array 0-11 evaluate_emfs_lhc40d_cims.sh  # done
sbatch --array 12-23 evaluate_emfs_lhc40d_cims.sh
# sbatch --array 24-107 evaluate_emfs_lhc40d_cims.sh  # done
sbatch --array 108-119 evaluate_emfs_lhc40d_cims.sh

sbatch --array 0-11 evaluate_emfa_lhc40d_cims.sh
# sbatch --array 12-59 evaluate_emfa_lhc40d_cims.sh  # done
sbatch --array 60-71 evaluate_emfa_lhc40d_cims.sh
# sbatch --array 72-83 evaluate_emfa_lhc40d_cims.sh  # done
sbatch --array 84-119 evaluate_emfa_lhc40d_cims.sh



# sbatch --array 0-119 evaluate_flow_scandal_lhc2d_cims.sh

# sbatch --array 0-119 evaluate_flow_scandal_lhc40d_cims.sh

# sbatch --array 0-119 evaluate_pie_scandal_lhc40d_cims.sh

# sbatch --array 0-119 evaluate_mfs_scandal_lhc40d_cims.sh

# sbatch --array 0-119 evaluate_mfa_scandal_lhc40d_cims.sh

# sbatch --array 0-119 evaluate_emfs_scandal_lhc40d_cims.sh

# sbatch --array 0-119 evaluate_emfa_scandal_lhc40d_cims.sh
