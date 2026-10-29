"""Aggregate closed-loop runs: <root>/<mode>/<model>/seed*/metrics.json -> all_runs.csv, summary.csv (mean ± std over seeds).

Usage: python simulation/summarize_closed.py --root results_sim/closed_2011
"""
import argparse, json, glob, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results_sim/closed_2011"); ROOT = ap.parse_args().root
rows = []
for f in glob.glob(ROOT + "/*/*/seed*/metrics.json"):
    p = f.split("/"); m = json.load(open(f)); m.update(mode=p[-4], model=p[-3], seed=p[-2]); rows.append(m)
df = pd.DataFrame(rows); df.to_csv(ROOT + "/all_runs.csv", index=False)
keys = ["cov_T", "width_T", "cov_RH", "width_RH", "rmse_T", "rmse_RH", "T_in_mean", "RH_in_mean", "RH_over85_frac", "vent_mean",
        "int_T_frac", "int_RH_frac", "eur_heat", "eur_rev", "eur_profit"]
g = df.groupby(["model", "mode"])[keys]
out = g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)
out["n_seeds"] = df.groupby(["model", "mode"]).size()
out.to_csv(ROOT + "/summary.csv"); print(out[["cov_T", "cov_RH", "width_T", "width_RH", "RH_over85_frac", "vent_mean", "n_seeds"]].to_string())
