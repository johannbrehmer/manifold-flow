#!/bin/bash


# sbatch timing.sh

# sbatch simulate.sh


# sbatch train_flow_spherical.sh
# sbatch train_pie_spherical.sh
# sbatch train_slice_spherical.sh
# sbatch --array=0-2 train_mf_spherical.sh
# sbatch train_gamf_spherical.sh
# sbatch train_hybrid_spherical.sh
# sbatch train_spie_spherical.sh
# sbatch train_smf_spherical.sh

# sbatch train_flow_csg.sh
# sbatch train_pie_csg.sh
# sbatch train_slice_csg.sh
# sbatch --array=0-2 train_mf_csg.sh
# sbatch train_gamf_csg.sh
# sbatch train_hybrid_csg.sh
# sbatch train_spie_csg.sh
# sbatch train_smf_csg.sh

# sbatch train_flow_tth.sh
# sbatch train_pie_tth.sh
# sbatch train_gamf_tth.sh
# sbatch train_mf_tth.sh
# sbatch train_slice_tth.sh
# sbatch train_hybrid_tth.sh

sbatch train_flow_tth2d.sh
sbatch train_pie_tth2d.sh
sbatch train_gamf_tth2d.sh
# sbatch train_mf_tth2d.sh
# sbatch train_hybrid_tth2d.sh
# sbatch train_slice_tth2d.sh


# sbatch evaluate_flow_spherical.sh
# sbatch evaluate_pie_spherical.sh
# sbatch evaluate_slice_spherical.sh
# sbatch evaluate_mf_spherical.sh
# sbatch evaluate_gamf_spherical.sh
# sbatch evaluate_hybrid_spherical.sh
# sbatch evaluate_spie_spherical.sh
# sbatch evaluate_smf_spherical.sh

# sbatch evaluate_flow_csg.sh
# sbatch evaluate_pie_csg.sh
# sbatch evaluate_slice_csg.sh
# sbatch evaluate_mf_csg.sh
# sbatch evaluate_gamf_csg.sh
# sbatch evaluate_hybrid_csg.sh
# sbatch evaluate_spie_csg.sh
# sbatch evaluate_smf_csg.sh

#sbatch evaluate_flow_tth.sh
#sbatch evaluate_pie_tth.sh
#sbatch evaluate_slice_tth.sh
#sbatch evaluate_mf_tth.sh
#sbatch evaluate_gamf_tth.sh
#sbatch evaluate_hybrid_tth.sh
