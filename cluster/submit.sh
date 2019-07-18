#!/usr/bin/env bash

sbatch --array=8-8 evaluate.sh
sbatch --array=16-16 evaluate.sh
sbatch --array=32-32 evaluate.sh
sbatch --array=64-64 evaluate.sh
sbatch --array=128-128 evaluate.sh
