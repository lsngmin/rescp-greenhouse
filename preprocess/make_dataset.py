"""10-min table -> modelling dataset: next-hour targets, valid-sample flag, chronological split.

  python make_dataset.py --name kangwon  --table kangwon_10min.parquet  --out data/
  python make_dataset.py --name agc1_ref --table agc1_ref_10min.parquet --out data/

Targets  T_in_max1h = max(T_in[t+10 .. t+60 min]) (all six values required), same for RH_in; T_in_h1, RH_in_h1 = value at t+60 min
valid    = the six input steps t-50..t have no NaN in the seven measured inputs and all four targets exist
split    = train / calib / test by the fixed dates below (train includes the 7-day validation period used by the trainers)
Output   <out>/<name>.parquet (nine inputs, four targets, valid, split) and <out>/splits.json
"""
import argparse, json, os, pandas as pd

SPLITS = {
 "agc1_ref": {"train": ("2018-08-14", "2018-10-25"), "calib": ("2018-10-26", "2018-11-08"), "test": ("2018-11-09", "2018-12-06")},
 "kangwon":  {"train": ("2023-07-27", "2023-10-07"), "calib": ("2023-10-08", "2023-10-21"), "test": ("2023-10-22", "2023-11-18")},
}
W = 6
INP = ["T_in", "RH_in", "T_out", "RH_out", "WS_out", "Light_out", "Vent"]

def main(name, table, out):
    df = pd.read_parquet(table)
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("10min")).all()
    tg = pd.DataFrame(index=df.index)
    for v in ["T_in", "RH_in"]:
        s = df[v]; tg[f"{v}_h1"] = s.shift(-6)
        tg[f"{v}_max1h"] = pd.concat([s.shift(-k) for k in range(1, 7)], axis=1).max(axis=1, skipna=False)
    ok_in = df[INP].notna().all(axis=1).astype(int).rolling(W).sum() == W
    valid = (ok_in & tg.notna().all(axis=1)).astype(int)
    full = df.join(tg); full["valid"] = valid; full["split"] = ""
    for k, (a, b) in SPLITS[name].items():
        m = (full.index >= pd.Timestamp(a)) & (full.index < pd.Timestamp(b) + pd.Timedelta(days=1)); full.loc[m, "split"] = k
    use = full[full.split != ""]
    os.makedirs(out, exist_ok=True); use.to_parquet(os.path.join(out, f"{name}.parquet"))
    sp = os.path.join(out, "splits.json"); allsp = json.load(open(sp)) if os.path.exists(sp) else {}
    allsp[name] = {k: {"start": a, "end": b} for k, (a, b) in SPLITS[name].items()}; json.dump(allsp, open(sp, "w"), indent=2)
    for k in SPLITS[name]:
        part = use[use.split == k]; print(f"{name} {k:5s} {len(part):5d} rows, {int(part.valid.sum()):5d} valid")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True, choices=list(SPLITS)); ap.add_argument("--table", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); main(a.name, a.table, a.out)
