#!/bin/bash
# Stage 3: GreenLight-Gym closed loop, Amsterdam 2011.
#   3a (env glgym): open-loop run of the rule-based controller -> results_sim/baseline_2011.csv
#   3b (env rescp): simulation dataset, samples and the seven forecasters -> data_sim/, results_sim/pred/sim_2011/
#   3c (env glgym): closed loop, 5 groups x 7 forecasters x 5 seeds -> results_sim/closed_2011/<mode>/<model>/seed<k>/
#   3d (env rescp): summary -> results_sim/closed_2011/{all_runs,summary}.csv
# One closed-loop run takes about 50-70 s; the 175 runs are executed with xargs -P.
set -e; cd "$(dirname "$0")/.."; ROOT=$PWD
NGPU=${NGPU:-2}; NPAR=${NPAR:-10}; mkdir -p logs/sim
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate glgym
python simulation/run_baseline.py --year 2011 --out results_sim/baseline_2011.csv > logs/sim/baseline.log 2>&1

conda activate rescp
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python simulation/make_sim_dataset.py --year 2011 --src results_sim/baseline_2011.csv --out data_sim
python forecasting/make_samples.py --data data_sim --out data_sim/samples --names sim_2011
python forecasting/run_lgbm.py --data data_sim --out results_sim/pred --name sim_2011 --trials 50 > logs/sim/pred_lgbm.log 2>&1 &
i=0
for m in gru xlstm moderntcn timemixer patchtst itransformer; do
  python forecasting/run_dl.py --data data_sim --out results_sim/pred --name sim_2011 --model $m --gpu $((i % NGPU)) --trials 50 > logs/sim/pred_${m}.log 2>&1 &
  i=$((i + 1))
done
wait

conda activate glgym
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
i=0
for mode in baseline D2 P P2 P3; do for m in lgbm gru xlstm moderntcn timemixer patchtst itransformer; do for s in 0 1 2 3 4; do
  echo "$m $s $mode $((i % NGPU))"; i=$((i + 1))
done; done; done | xargs -P $NPAR -L 1 bash -c \
  "cd $ROOT; [ -f results_sim/closed_2011/\$2/\$0/seed\$1/metrics.json ] || python simulation/run_closed_loop.py --model \$0 --seed \$1 --mode \$2 --gpu \$3 --year 2011 --data data_sim --results results_sim > logs/sim/closed_\$2_\$0_s\$1.log 2>&1 || echo FAILED \$0 \$1 \$2"

conda activate rescp
python simulation/summarize_closed.py --root results_sim/closed_2011
echo "[$(date)] simulation done"
