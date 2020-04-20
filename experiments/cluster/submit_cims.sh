#!/bin/bash

# Last update master: April 15, 9:40
# Last update scandal: April 19, 13:40



sbatch --array 0-119 evaluate_flow_lhc2d_cims.sh  # done
sbatch --array 0-119 evaluate_flow_lhc40d_cims.sh  # done
sbatch --array 0-119 evaluate_pie_lhc40d_cims.sh  # done
# sbatch --array 0-119 evaluate_pie_cm_lhc40d_cims.sh
sbatch --array 0-119 evaluate_mfs_lhc40d_cims.sh  # done
sbatch --array 0-119 evaluate_mfa_lhc40d_cims.sh  # done
sbatch --array 0-119 evaluate_emfs_lhc40d_cims.sh  # done
sbatch --array 0-119 evaluate_emfa_lhc40d_cims.sh  # done



sbatch --array 0-47 evaluate_flow_scandal_lhc2d_cims.sh  # done
# sbatch --array 48-119 evaluate_flow_scandal_lhc2d_cims.sh

sbatch --array 0-47 evaluate_flow_scandal_lhc40d_cims.sh  # done
# sbatch --array 48-119 evaluate_flow_scandal_lhc40d_cims.sh

sbatch --array 0-47 evaluate_pie_scandal_lhc40d_cims.sh  # done
# sbatch --array 48-119 evaluate_pie_scandal_lhc40d_cims.sh

# sbatch --array 0-119 evaluate_pie_cm_scandal_lhc40d_cims.sh

sbatch --array 0-59 evaluate_mfs_scandal_lhc40d_cims.sh  # done
# sbatch --array 60-119 evaluate_mfs_scandal_lhc40d_cims.sh

# sbatch --array 0-119 evaluate_mfa_scandal_lhc40d_cims.sh

sbatch --array 0-35 evaluate_emfs_scandal_lhc40d_cims.sh  # done
# sbatch --array 36-119 evaluate_emfs_scandal_lhc40d_cims.sh

# sbatch --array 0-11 evaluate_emfa_scandal_lhc40d_cims.sh
sbatch --array 12-23 evaluate_emfa_scandal_lhc40d_cims.sh  # done
# sbatch --array 24-35 evaluate_emfa_scandal_lhc40d_cims.sh
sbatch --array 36-47 evaluate_emfa_scandal_lhc40d_cims.sh  # done
# sbatch --array 48-119 evaluate_emfa_scandal_lhc40d_cims.sh
