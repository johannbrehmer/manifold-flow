#!/bin/bash


# sbatch timing.sh

# sbatch simulate.sh


# sbatch train_flow_spherical.sh  # some missing
# sbatch train_pie_spherical.sh  # some missing
# sbatch train_gamf_spherical.sh  # some missing
# sbatch --array=0-2 train_spie_spherical.sh  # not yet done
# sbatch --array=0-2 train_smf_spherical.sh  # not yet done (NaNs in likelihood -- fixed now?)
# sbatch --array=0-2 train_mf_spherical.sh  # not yet done
# sbatch train_pie_epsilon_spherical.sh  # not yet done, wait for L2 results

# sbatch train_flow_csg.sh  # some missing
# sbatch train_pie_csg.sh  # some missing
# sbatch train_gamf_csg.sh  # some missing
# sbatch --array=0-2 train_spie_csg.sh  # not yet done
# sbatch --array=0-2 train_smf_csg.sh  # not yet done (NaNs in likelihood -- fixed now?)
# sbatch --array=0-2 train_mf_csg.sh  # not yet done
# sbatch train_pie_epsilon_csg.sh  # not yet done, wait for L2 results

# sbatch train_flow_tth2d.sh  # done
# sbatch train_flow_tth.sh  # done
# sbatch train_pie_tth.sh  # done
# sbatch train_gamf_tth.sh  # done
# sbatch train_mf_tth.sh  # CUDA memory issue, next try with bs 50


sbatch evaluate_flow_spherical.sh  # not yet done, some not yet trained
sbatch evaluate_pie_spherical.sh  # not yet done, some not yet trained
sbatch evaluate_gamf_spherical.sh  # not yet done, some not yet trained
# sbatch evaluate_spie_spherical.sh  # not yet trained
# sbatch evaluate_smf_spherical.sh  # not yet trained
# sbatch evaluate_mf_spherical.sh  # not yet trained
# sbatch evaluate_pie_epsilon_spherical.sh  # not yet trained

sbatch evaluate_flow_csg.sh  # not yet done, some not yet trained
sbatch evaluate_pie_csg.sh  # not yet done, some not yet trained
sbatch evaluate_gamf_csg.sh  # not yet done, some not yet trained
# sbatch evaluate_spie_csg.sh  # not yet trained
# sbatch evaluate_smf_csg.sh  # not yet trained
# sbatch evaluate_mf_csg.sh  # not yet trained
# sbatch evaluate_pie_epsilon_csg.sh  # not yet trained

sbatch evaluate_flow_tth2d.sh  # not yet done
sbatch evaluate_flow_tth.sh  # not yet done
sbatch evaluate_pie_tth.sh  # not yet done
sbatch evaluate_gamf_tth.sh  # not yet done
# sbatch evaluate_mf_tth.sh  # not yet trained
