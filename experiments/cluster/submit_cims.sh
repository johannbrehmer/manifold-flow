#!/bin/bash

sbatch --array 0-9 evaluate_flow_lhc2d_cims.sh
sbatch --array 0-9 evaluate_flow_lhc_cims.sh
sbatch --array 0-9 evaluate_pie_lhc_cims.sh
sbatch --array 0-9 evaluate_mf_lhc_cims.sh
sbatch --array 0-9 evaluate_emf_lhc_cims.sh
sbatch --array 0-19 evaluate_gamf_lhc_cims.sh
