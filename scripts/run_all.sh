#!/bin/bash
# Full pipeline after the processed datasets are in data/ (see README, Data). Stages 1, 2 and 4 run in env rescp;
# stage 3 switches between glgym and rescp itself.
set -e; cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rescp
bash scripts/run_01_forecast.sh
bash scripts/run_02_conformal.sh
bash scripts/run_03_simulation.sh
conda activate rescp
bash scripts/run_04_figures.sh
echo "[$(date)] all done"
