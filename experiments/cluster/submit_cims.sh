#!/bin/bash

sbatch --array 1-2 evaluate_flow_lhc2d_cims.sh
sbatch --array 1-2 evaluate_flow_lhc_cims.sh
sbatch --array 1-2 evaluate_pie_lhc_cims.sh
sbatch --array 1-2 evaluate_mf_lhc_cims.sh
sbatch --array 1-2 evaluate_emf_lhc_cims.sh
sbatch --array 2-5 evaluate_gamf_lhc_cims.sh
