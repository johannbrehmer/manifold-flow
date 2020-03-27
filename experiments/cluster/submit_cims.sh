#!/bin/bash


############################################################
# Run 0
############################################################

# sbatch --array 0-11 evaluate_flow_lhc2d_cims.sh  # done

# sbatch --array 0-11 evaluate_flow_lhc40d_cims.sh  # submitted
# sbatch --array 0-11 evaluate_pie_lhc40d_cims.sh  # submitted
# sbatch --array 0-11 evaluate_mf_lhc40d_cims.sh
sbatch --array 0-11 evaluate_emf_lhc40d_cims.sh

# sbatch --array 0-11 evaluate_flow_lhc_cims.sh  # done
# sbatch --array 0-11 evaluate_pie_lhc_cims.sh  # done
# sbatch --array 0-11 evaluate_mf_lhc_cims.sh  # done
# sbatch --array 0-11 evaluate_emf_lhc_cims.sh  # done

# sbatch --array 0-11 evaluate_flow_lhc_long_cims.sh  # submitted
# sbatch --array 0-11 evaluate_pie_lhc_long_cims.sh  # submitted
# sbatch --array 0-11 evaluate_mf_lhc_long_cims.sh
# sbatch --array 0-11 evaluate_emf_lhc_long_cims.sh


# sleep 60


############################################################
# Run 1-2
############################################################

# sbatch --array 12-35 evaluate_flow_lhc2d_cims.sh  # done

# sbatch --array 12-35 evaluate_flow_lhc40d_cims.sh  # submitted
# sbatch --array 12-35 evaluate_pie_lhc40d_cims.sh  # submitted
# sbatch --array 12-35 evaluate_mf_lhc40d_cims.sh
sbatch --array 12-23 evaluate_emf_lhc40d_cims.sh
# sbatch --array 24-35 evaluate_emf_lhc40d_cims.sh

# sbatch --array 12-35 evaluate_flow_lhc_cims.sh  # done
# sbatch --array 12-35 evaluate_pie_lhc_cims.sh  # done
# sbatch --array 12-35 evaluate_mf_lhc_cims.sh  # done
# sbatch --array 12-35 evaluate_emf_lhc_cims.sh  # done

# sbatch --array 12-35 evaluate_flow_lhc_long_cims.sh
# sbatch --array 12-35 evaluate_pie_lhc_long_cims.sh
# sbatch --array 12-35 evaluate_mf_lhc_long_cims.sh
# sbatch --array 12-35 evaluate_emf_lhc_long_cims.sh


# sleep 60


############################################################
# Run 3-4
############################################################

# sbatch --array 36-59 evaluate_flow_lhc2d_cims.sh  # done

# sbatch --array 36-59 evaluate_flow_lhc40d_cims.sh  # submitted
# sbatch --array 36-59 evaluate_pie_lhc40d_cims.sh  # submitted
# sbatch --array 36-59 evaluate_mf_lhc40d_cims.sh
sbatch --array 36-47 evaluate_emf_lhc40d_cims.sh
# sbatch --array 48-59 evaluate_emf_lhc40d_cims.sh

# sbatch --array 36-59 evaluate_flow_lhc_cims.sh  # done
# sbatch --array 36-59 evaluate_pie_lhc_cims.sh  # done
# sbatch --array 36-59 evaluate_mf_lhc_cims.sh  # done
sbatch --array 36-47 evaluate_emf_lhc_cims.sh
# sbatch --array 48-59 evaluate_emf_lhc_cims.sh  # submitted

# sbatch --array 36-59 evaluate_flow_lhc_long_cims.sh
# sbatch --array 36-59 evaluate_pie_lhc_long_cims.sh
# sbatch --array 36-59 evaluate_mf_lhc_long_cims.sh
# sbatch --array 36-59 evaluate_emf_lhc_long_cims.sh
