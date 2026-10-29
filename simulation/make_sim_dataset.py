"""Open-loop trace -> modelling dataset in the same format as the measured datasets
(nine inputs + four targets + valid + split). Same calendar split as the Netherlands dataset.

Usage: python simulation/make_sim_dataset.py --year 2011 --src results_sim/baseline_2011.csv --out data_sim
"""
import argparse, json
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--year", type=int, default=2011); ap.add_argument("--src", default="results_sim/baseline_2011.csv"); ap.add_argument("--out", default="data_sim")
a = ap.parse_args(); YEAR = a.year; NAME = f"sim_{YEAR}"; out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(a.src, index_col=0, parse_dates=True)
INP = ["T_in", "RH_in", "T_out", "RH_out", "WS_out", "Light_out", "Vent", "hour_sin", "hour_cos"]
out = df[INP].copy()
for v in ["T_in", "RH_in"]:
    s = out[v]; out[f"{v}_h1"] = s.shift(-6)
    out[f"{v}_max1h"] = pd.concat([s.shift(-k) for k in range(1, 7)], axis=1).max(axis=1, skipna=False)
ok_in = out[INP].notna().all(axis=1).astype(int).rolling(6).sum() == 6
out["valid"] = (ok_in & out[["T_in_h1", "RH_in_h1", "T_in_max1h", "RH_in_max1h"]].notna().all(axis=1)).astype(int)
S = {"train": (f"{YEAR}-08-14", f"{YEAR}-10-25"), "calib": (f"{YEAR}-10-26", f"{YEAR}-11-08"), "test": (f"{YEAR}-11-09", f"{YEAR}-12-06")}
out["split"] = None
for k, (s0, s1) in S.items():
    m = (out.index >= s0) & (out.index < pd.Timestamp(s1) + pd.Timedelta(days=1)); out.loc[m, "split"] = k
out.index.name = "ts"; out.to_parquet(out_dir / f"{NAME}.parquet")
json.dump(S, open(out_dir / f"splits_{NAME}.json", "w"))
for k in S: m = out.split == k; print(k, int(m.sum()), "valid", int(out.valid[m].sum()))
