#!/bin/bash
# Stage 1: samples + seven forecasters on the two measured datasets (conda env rescp).
# Expects data/kangwon.parquet and data/agc1_ref.parquet (Zenodo or preprocess/). Writes results/<dataset>/<model>/.
# Tuning takes roughly 1.5 min (LightGBM) to 50 min (iTransformer) per dataset on one A40; GPU jobs are spread over two GPUs.
set -e; cd "$(dirname "$0")/.."
NGPU=${NGPU:-2}; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; mkdir -p logs/forecast
python forecasting/make_samples.py --data data --out data/samples --names kangwon agc1_ref
for ds in kangwon agc1_ref; do
  python forecasting/run_lgbm.py --data data --out results --name $ds --trials 50 > logs/forecast/${ds}_lgbm.log 2>&1 &
  i=0
  for m in gru xlstm moderntcn timemixer patchtst itransformer; do
    python forecasting/run_dl.py --data data --out results --name $ds --model $m --gpu $((i % NGPU)) --trials 50 > logs/forecast/${ds}_${m}.log 2>&1 &
    i=$((i + 1))
  done
  wait
done
echo "[$(date)] forecasting done"
