"""Open-loop run of the rule-based controller for one season (conda env glgym).
Writes the 10-min trace (nine variables + set points) used to build the simulation dataset.

Usage: python simulation/run_baseline.py --year 2011 --out results_sim/baseline_2011.csv
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd
from sim_common import make, ctx_of, record, RuleBasedController

ap = argparse.ArgumentParser()
ap.add_argument("--year", type=int, default=2011); ap.add_argument("--out", default="results_sim/baseline_2011.csv")
a = ap.parse_args(); out = Path(a.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
env, rb, scen = make(year=a.year); ctl = RuleBasedController(**rb)
t0 = time.time(); env.reset(seed=0, options={"scenario": scen})
ts0 = pd.Timestamp(f"{a.year}-01-01") + pd.Timedelta(days=scen["start_day"] - 1)
rows = []; tot = {"heat": 0., "elec": 0., "co2": 0., "rev": 0., "profit": 0.}
while True:
    u = ctl.predict(ctx_of(env)); _, r, term, trunc, info = env.step(u.astype(np.float32))
    rec = record(env, ctl); rec["ts"] = ts0 + pd.Timedelta(seconds=env.dt * env.timestep); rows.append(rec)
    tot["heat"] += info["heat_cost"]; tot["elec"] += info["elec_cost"]; tot["co2"] += info["co2_cost"]; tot["rev"] += info["revenue"]; tot["profit"] += info["profit"]
    if term or trunc: break
df = pd.DataFrame(rows).set_index("ts"); df.to_csv(out)
print("steps", len(df), "sec", round(time.time() - t0, 1), "period", df.index.min(), "~", df.index.max())
print(df.describe().T[["mean", "std", "min", "max"]].round(2).to_string())
print("frac T_in > vent set point", round((df.T_in > df.T_vent_sp).mean(), 3), "| frac RH_in > 85", round((df.RH_in > 85).mean(), 3))
print("cumulative (EUR/m2)", {k: round(v, 2) for k, v in tot.items()})
