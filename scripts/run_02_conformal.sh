#!/bin/bash
# Stage 2: ResCP (T = 0.3, buffer 2000) and the two baselines (sliding-window CP, ACI) on all 14 dataset-model pairs (conda env rescp).
# Reads results/<dataset>/<model>/seed*/pred.csv, writes rescp_*.csv / base_*.csv and the summary JSON files next to them.
set -e; cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2; mkdir -p logs/conformal
for ds in kangwon agc1_ref; do
  for m in lgbm gru xlstm moderntcn timemixer patchtst itransformer; do
    python conformal/run_rescp.py --results results --name $ds --model $m --T 0.3 --window 2000 --gpu 0 > logs/conformal/${ds}_${m}_rescp.log 2>&1
    python conformal/baselines_cp.py --results results --name $ds --model $m > logs/conformal/${ds}_${m}_base.log 2>&1
    echo "$ds $m done"
  done
done
echo "[$(date)] conformal done"
