#!/bin/bash

################################################################################
# IMDb training jobs
################################################################################

sbatch --array 0-0 train_mf_imdb_bnl.sh
sbatch --array 0-0 train_flow_imdb_bnl.sh
# sbatch --array 0-0 train_emf_imdb_bnl.sh
# sbatch --array 0-0 train_pie_imdb_bnl.sh
