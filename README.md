# rescp-greenhouse

Code for *Reservoir Conformal Prediction for Greenhouse Climate Forecasting under Seasonal and Control-Induced Distribution Shifts* (IEEE Access, under review).

Prediction intervals for the next-hour maximum indoor air temperature and relative humidity in greenhouses, built with ResCP on seven base forecasters, evaluated on two real greenhouse datasets (Korea, Netherlands) and in closed-loop simulation with GreenLight-Gym.

Archived versions: code https://doi.org/10.5281/zenodo.23754280, data and results https://doi.org/10.5281/zenodo.23754795 (processed datasets, trained models, prediction intervals, closed-loop traces and figures).

## Layout

| Folder | Contents |
|---|---|
| `preprocess/` | raw downloads -> `data/{kangwon,agc1_ref}.parquet` (10-min, nine inputs, four targets, valid flag, split) |
| `forecasting/` | samples (`make_samples.py`), LightGBM (`run_lgbm.py`) and six deep forecasters (`run_dl.py`); shared code in `common.py` |
| `conformal/` | ResCP (`run_rescp.py`) and the two baselines, sliding-window CP and ACI (`baselines_cp.py`) |
| `simulation/` | GreenLight-Gym open-loop run, simulation dataset, closed-loop runs and their summary; `configs/envs/GreenLightEnv.yml` |
| `figures/` | `make_figures.py`: all figures and the 28-condition baseline table |
| `scripts/` | the four stages in order and `run_all.sh` |
| `third_party/` | git submodules at the commits used (see Setup) |

Result layout: `results/<dataset>/<model>/seed<k>/` holds `pred.csv` (point forecasts), `rescp_{T,RH}_a90_w2000_T0.3.csv` and `base_{swcp,aci}_{T,RH}.csv` (intervals), with per-model summaries `rescp_summary_a90_w2000_T0.3.json` and `base_summary.json`. Closed-loop runs are in `results_sim/closed_2011/<mode>/<model>/seed<k>/{trace.csv,metrics.json}` with `mode` in `baseline`, `D2` (interval-based control; "interval control" in the paper), `P`, `P2`, `P3` (point-forecast control with cold margin delta = 0, 1.0, 1.5 degC).

## Setup

Two conda environments: `rescp` for forecasting, conformal prediction and figures, and `glgym` for the GreenLight-Gym closed-loop simulation. They are separate because `gl-gym` pins a different numpy than the forecasting stack.

```bash
conda env create -f environment_rescp.yml
conda env create -f environment_glgym.yml
git submodule update --init
```

The submodules are pinned to the commits used for the paper:

| Package | Repository | Commit | Used by |
|---|---|---|---|
| ResCP | https://github.com/RobertoNeglia/reservoir-conformal-prediction | 1d8e560 | `conformal/`, `simulation/` |
| ModernTCN | https://github.com/luodhhh/ModernTCN | 56a9a2c | `forecasting/run_dl.py` |
| Time-Series-Library | https://github.com/thuml/Time-Series-Library | 4e938a1 | `forecasting/run_dl.py` (PatchTST, iTransformer, TimeMixer) |
| GreenLight-Gym | https://github.com/BartvLaatum/GreenLight-Gym | 2d3febb | `simulation/` (weather files, rule-based controller config) |

Closed-loop results were obtained in the `glgym` environment as exported here; simulator outputs can differ numerically in other environments.

## Data

- Korea dataset: Mendeley Data, doi:10.17632/mwg293k2r9.2 (CC BY 4.0). Outdoor temperature, humidity and wind speed are taken from KMA ASOS station 211 (Inje) 1-min observations, downloaded as CSV from the KMA Open MET Data Portal (data.kma.go.kr, KOGL Type 1).
- Netherlands dataset: Autonomous Greenhouse Challenge, first edition, 4TU.ResearchData, doi:10.4121/uuid:e4987a7b-04dd-4c89-9b18-883aad30ba9a (CC0). Reference compartment.
- Simulation weather: Amsterdam 2011, included in GreenLight-Gym.

Processed files (`data/kangwon.parquet`, `data/agc1_ref.parquet`, `data/splits.json`) are in `data.zip` of the Zenodo data record (https://doi.org/10.5281/zenodo.23754795). To rebuild them from the raw downloads:

```bash
python preprocess/preprocess_kangwon.py --raw raw/kangwon/HTVT_validation_10min.csv \
    --kma raw/kma/SURFACE_ASOS_211_MI_2023.csv --out data/kangwon_10min.parquet
python preprocess/preprocess_agc1.py --climate "raw/agc1/Reference(Growers)/Greenhouse_climate.csv" \
    --meteo raw/agc1/meteo.csv --out data/agc1_ref_10min.parquet
python preprocess/make_dataset.py --name kangwon  --table data/kangwon_10min.parquet  --out data/
python preprocess/make_dataset.py --name agc1_ref --table data/agc1_ref_10min.parquet --out data/
```

## Run

With `data/` in place:

```bash
bash scripts/run_all.sh
```

| Stage | Script | Env | Output |
|---|---|---|---|
| 1 | `scripts/run_01_forecast.sh` | rescp | `data/samples/`, `results/<dataset>/<model>/` |
| 2 | `scripts/run_02_conformal.sh` | rescp | ResCP and baseline intervals next to `pred.csv` |
| 3 | `scripts/run_03_simulation.sh` | glgym + rescp | `results_sim/baseline_2011.csv`, `data_sim/`, `results_sim/pred/sim_2011/`, `results_sim/closed_2011/` |
| 4 | `scripts/run_04_figures.sh` | rescp | `figs/` |

Each stage can be run on its own; stage 3 switches environments itself. The scripts assume two GPUs (`NGPU`) and run up to `NPAR` closed-loop runs in parallel. Stage 1 is the slowest (hyper-parameter tuning, up to about 50 min per deep forecaster per dataset on one A40); the full pipeline takes several hours. To reproduce only the figures from the published results, unzip `data.zip`, `results.zip` and `results_sim.zip` from the Zenodo data record in the repository root and run stage 4.

## License

AGPL-3.0, following GreenLight-Gym. See `LICENSE`. The third-party packages keep their own licenses (ResCP: MIT; ModernTCN and Time-Series-Library: MIT; GreenLight-Gym: AGPL-3.0).
