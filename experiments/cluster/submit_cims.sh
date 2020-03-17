#!/bin/bash

sbatch --array 15-24 evaluate_flow_lhc2d_cims.sh
sbatch --array 15-24 evaluate_flow_lhc_cims.sh
sbatch --array 15-24 evaluate_pie_lhc_cims.sh
sbatch --array 15-24 evaluate_mf_lhc_cims.sh
sbatch --array 15-24 evaluate_emf_lhc_cims.sh
sbatch --array 30-49 evaluate_gamf_lhc_cims.sh
