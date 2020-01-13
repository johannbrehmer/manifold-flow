#!/bin/bash


# sbatch timing.sh

# sbatch simulate.sh


# sbatch train_flow_spherical.sh
# sbatch train_pie_spherical.sh
# sbatch train_gamf_spherical.sh
# sbatch --array=0-2 train_spie_spherical.sh  # Missing
# sbatch --array=0-2 train_smf_spherical.sh  # NaNs appear
# sbatch --array=0-2 train_mf_spherical.sh  # Missing

# sbatch train_flow_csg.sh
# sbatch train_pie_csg.sh
# sbatch train_gamf_csg.sh
# sbatch --array=0-2 train_spie_csg.sh  # Missing
# sbatch --array=0-2 train_smf_csg.sh  # NaNs appear
# sbatch --array=0-2 train_mf_csg.sh  # Missing

# sbatch train_flow_tth2d.sh
# sbatch train_flow_tth.sh
# sbatch train_pie_tth.sh
# sbatch train_gamf_tth.sh
# sbatch train_mf_tth.sh  # CUDA memory issue


sbatch evaluate_flow_spherical.sh
sbatch evaluate_pie_spherical.sh
sbatch evaluate_gamf_spherical.sh
# sbatch evaluate_spie_spherical.sh
# sbatch evaluate_smf_spherical.sh
# sbatch evaluate_mf_spherical.sh

sbatch evaluate_flow_csg.sh
sbatch evaluate_pie_csg.sh
sbatch evaluate_gamf_csg.sh
# sbatch evaluate_spie_csg.sh
# sbatch evaluate_smf_csg.sh
# sbatch evaluate_mf_csg.sh

sbatch evaluate_flow_tth.sh
sbatch evaluate_pie_tth.sh
sbatch evaluate_gamf_tth.sh
# sbatch evaluate_mf_tth.sh
