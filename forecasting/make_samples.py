"""
Build model samples: data/{name}.parquet -> data/samples/{name}_samples.npz

Input   X[i] : past W=6 steps (t-50 min .. t, inclusive) x 9 variables -> (N, 6, 9)
Target  y[i] : [T_in_max1h, RH_in_max1h] at t                          -> (N, 2)
               T_in_max1h = max(T_in[t+10 min .. t+60 min]) (already computed in the parquet)
Valid   all 54 input values and both targets are non-NaN; no imputation
Split   the split column of the parquet (train / calib / test)

Usage: python make_samples.py --data data --out data/samples
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

INP = ["T_in", "RH_in", "T_out", "RH_out", "WS_out", "Light_out", "Vent", "hour_sin", "hour_cos"]
TGT = ["T_in_max1h", "RH_in_max1h"]
W = 6
STEP = pd.Timedelta(minutes=10)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # grid check: continuous 10-min steps, no duplicates
    d = np.diff(df.index.values).astype("timedelta64[m]").astype(int)
    assert (d == 10).all(), f"not a 10-min grid: unique diffs={np.unique(d)}"
    assert df.index.is_unique
    missing = [c for c in INP + TGT + ["split"] if c not in df.columns]
    assert not missing, f"missing columns: {missing}"
    return df


def build(df: pd.DataFrame):
    Xall = df[INP].to_numpy(dtype=np.float32)          # (T, 9)
    yall = df[TGT].to_numpy(dtype=np.float32)          # (T, 2)
    T = len(df)
    # window for row i: Xall[i-W+1 : i+1]  (i >= W-1)
    idx = np.arange(W - 1, T)
    win = idx[:, None] + np.arange(-W + 1, 1)[None, :]  # (N, W)
    X = Xall[win]                                      # (N, W, 9)
    y = yall[idx]                                      # (N, 2)
    ts = df.index.values[idx]
    split = df["split"].to_numpy()[idx]
    valid = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(y).all(axis=1)
    return X, y, ts, split, valid, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/samples")
    ap.add_argument("--names", nargs="+", default=["kangwon", "agc1_ref"])
    a = ap.parse_args()
    data, out = Path(a.data).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    meta_all = {}
    for name in a.names:
        df = load(data / f"{name}.parquet")
        X, y, ts, split, valid, idx = build(df)

        # cross-check against the valid column of the parquet (if present)
        if "valid" in df.columns:
            v_file = df["valid"].to_numpy()[idx].astype(bool)
            mism = int((v_file != valid).sum())
            print(f"[{name}] recomputed valid vs file: {mism} mismatching rows")
            if mism:
                bad = np.where(v_file != valid)[0][:10]
                print("   e.g.:", [(str(ts[b])[:16], bool(v_file[b]), bool(valid[b])) for b in bad])

        labels = [s for s in ["train", "calib", "test"] if s in set(split)]
        unknown = set(split) - set(labels) - {None, "", np.nan}
        unknown = {u for u in unknown if isinstance(u, str)}
        assert not unknown, f"unknown split labels: {unknown}"

        arrays, meta = {}, {"inputs": INP, "targets": TGT, "window": W, "splits": {}}
        for s in labels:
            m = (split == s) & valid
            arrays[f"X_{s}"] = X[m]
            arrays[f"y_{s}"] = y[m]
            arrays[f"ts_{s}"] = ts[m].astype("datetime64[ns]")
            n_all = int((split == s).sum())
            meta["splits"][s] = {
                "rows": n_all,
                "valid": int(m.sum()),
                "start": str(ts[split == s].min())[:16],
                "end": str(ts[split == s].max())[:16],
            }
            print(f"[{name}] {s:5s} rows {n_all:6d}  valid {int(m.sum()):6d}  "
                  f"{meta['splits'][s]['start']} ~ {meta['splits'][s]['end']}")
            # split boundary: number of samples whose window reaches into the previous split (at most the first W-1)
            first = np.where(split == s)[0][: W - 1]
            cross = int(sum(1 for f in first if f - W + 1 >= 0 and split[f - W + 1] != s))
            meta["splits"][s]["window_crosses_prev_split"] = cross

        # basic target statistics (valid samples, per split)
        for s in labels:
            ys = arrays[f"y_{s}"]
            meta["splits"][s]["y_mean"] = [round(float(v), 3) for v in ys.mean(0)]
            meta["splits"][s]["y_max"] = [round(float(v), 3) for v in ys.max(0)]

        np.savez_compressed(out / f"{name}_samples.npz", **arrays)
        meta_all[name] = meta
        print(f"[{name}] X shape {X.shape} -> saved {out / (name + '_samples.npz')}")

    (out / "samples_meta.json").write_text(json.dumps(meta_all, ensure_ascii=False, indent=2))
    print("meta ->", out / "samples_meta.json")


if __name__ == "__main__":
    main()
