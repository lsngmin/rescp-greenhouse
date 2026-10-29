"""Netherlands (Autonomous Greenhouse Challenge 2018, Reference compartment) -> 10-min table.

Inputs
  --climate  Reference(Growers)/Greenhouse_climate.csv  (5-min, GHtime = Excel serial)
  --meteo    meteo.csv                                   (5-min)
Output
  --out      parquet with the same nine columns as the Korea dataset

Processing
  * GHtime (Excel serial) -> datetime, rounded to 1 min
  * meteo.csv in the public copy carries dates only; row count and per-day row counts are checked
    against the climate file and timestamps are restored by position
  * 5-min -> 10-min by taking the value at each 10-min mark (no averaging)
  * vent = leeward vent (VentLee)
"""
import argparse, numpy as np, pandas as pd

def xl2ts(x): return pd.to_datetime("1899-12-30") + pd.to_timedelta(pd.to_numeric(x, errors="coerce"), unit="D")

def hour_features(ts):
    minute = ts.hour * 60 + ts.minute
    return np.sin(2 * np.pi * minute / 1440), np.cos(2 * np.pi * minute / 1440)

def main(climate, meteo, out):
    g = pd.read_csv(climate); ts = xl2ts(g["GHtime"]).dt.round("1min")
    assert ts.is_monotonic_increasing and not ts.duplicated().any()
    g.index = ts; g.index.name = "ts"
    m = pd.read_csv(meteo); md = pd.to_datetime(m["time"], errors="coerce")
    assert len(m) == len(g)
    assert (md.dt.normalize().values == ts.dt.normalize().values).all(), "meteo rows do not align with climate rows"
    m.index = ts.values; m.index.name = "ts"

    grid = pd.date_range(g.index.min().ceil("10min"), g.index.max().floor("10min"), freq="10min")
    num = lambda s: pd.to_numeric(s, errors="coerce").reindex(grid)
    hs, hc = hour_features(grid)
    res = pd.DataFrame({"T_in": num(g["Tair"]), "RH_in": num(g["RHair"]),
                        "T_out": num(m["Tout"]), "RH_out": num(m["Rhout"]), "WS_out": num(m["Windsp"]),
                        "Light_out": num(m["Iglob"]), "Vent": num(g["VentLee"]), "hour_sin": hs, "hour_cos": hc}, index=grid)
    res.index.name = "ts"
    res.to_parquet(out); print(f"{out}: {len(res)} rows, {res.index.min()} ~ {res.index.max()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--climate", required=True); ap.add_argument("--meteo", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); main(a.climate, a.meteo, a.out)
