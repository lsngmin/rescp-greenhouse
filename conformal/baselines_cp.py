"""
Two interval baselines with the same 6-step delay and the same buffer as ResCP, without similarity weighting.
  swcp : sliding-window split CP. Quantiles of the signed residuals in the buffer (last W observed residuals),
         minimum-width interval over the same beta grid as ResCP. Equals ResCP with similarity weights and time decay removed.
  aci  : adaptive conformal inference (Gibbs & Candes, 2021). The miscoverage level a_t is updated with the
         miscoverage indicator that arrives 6 steps late: a_{t+1} = a_t + gamma (alpha - err_{t-6}).
         Symmetric interval from the (1 - a_t) quantile of absolute residuals in the buffer. gamma = 0.01, a_t clipped to [0.005, 0.5].

Input   results/<name>/<model>/seed<k>/pred.csv
Output  same folder: base_<method>_<target>.csv, base_metrics.json; model folder: base_summary.json

Usage: python baselines_cp.py --name kangwon --model lgbm [--results results]
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
LAG, W, ALPHA, NQ, GAMMA = 6, 2000, 0.1, 100, 0.01
TARGETS = {"T": ("y_T", "p_T"), "RH": ("y_RH", "p_RH")}

def swcp(y, p, n_cal):
    r = y - p; N = len(y); lo = np.full(N - n_cal, np.nan); hi = lo.copy()
    lb = np.linspace(1e-3, ALPHA - 1e-3, NQ); ub = 1 - ALPHA + lb
    for k in range(n_cal, N):
        avail = k - LAG; buf = r[max(0, avail - W + 1):avail + 1]
        ql = np.quantile(buf, lb); qu = np.quantile(buf, ub); i = np.argmin(qu - ql)
        lo[k - n_cal] = p[k] + ql[i]; hi[k - n_cal] = p[k] + qu[i]
    return lo, hi

def aci(y, p, n_cal):
    r = y - p; N = len(y); lo = np.full(N - n_cal, np.nan); hi = lo.copy()
    a = ALPHA; alphas = []
    err = {}  # k -> miscoverage indicator of prediction issued at k (known at k+LAG)
    for k in range(n_cal, N):
        if k - LAG in err: a = float(np.clip(a + GAMMA * (ALPHA - err[k - LAG]), 0.005, 0.5))
        avail = k - LAG; buf = np.abs(r[max(0, avail - W + 1):avail + 1])
        q = np.quantile(buf, min(1.0, np.ceil((len(buf) + 1) * (1 - a)) / len(buf)))
        lo[k - n_cal] = p[k] - q; hi[k - n_cal] = p[k] + q
        err[k] = float(not (lo[k - n_cal] <= y[k] <= hi[k - n_cal])); alphas.append(a)
    return lo, hi, np.array(alphas)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--name", required=True); ap.add_argument("--model", required=True)
    a = ap.parse_args(); name, model = a.name, a.model
    root = Path(a.results).expanduser() / name / model
    summ = {}
    for sd in sorted(root.glob("seed*")):
        df = pd.read_csv(sd / "pred.csv", parse_dates=["ts"]); n_cal = int((df.split == "calib").sum())
        te = df.iloc[n_cal:].reset_index(drop=True); met = {}
        for tg, (yc, pc) in TARGETS.items():
            y, p = df[yc].to_numpy(), df[pc].to_numpy()
            for meth in ["swcp", "aci"]:
                if meth == "swcp": lo, hi = swcp(y, p, n_cal); extra = {}
                else: lo, hi, al = aci(y, p, n_cal); extra = {"alpha_mean": float(al.mean()), "alpha_min": float(al.min()), "alpha_max": float(al.max())}
                out = pd.DataFrame({"ts": te.ts, "y": te[yc], "p": te[pc], "lo": lo, "hi": hi}); out.to_csv(sd / f"base_{meth}_{tg}.csv", index=False)
                met[f"{meth}_{tg}"] = {"coverage": float(((out.y >= out.lo) & (out.y <= out.hi)).mean()), "width": float((out.hi - out.lo).mean()), **extra}
        (sd / "base_metrics.json").write_text(json.dumps(met, indent=2)); summ[sd.name] = met
    keys = list(next(iter(summ.values())).keys())
    agg = {k: {m: [float(np.mean([summ[s][k][m] for s in summ])), float(np.std([summ[s][k][m] for s in summ]))] for m in ["coverage", "width"]} for k in keys}
    (root / "base_summary.json").write_text(json.dumps({"seeds": summ, "mean_std": agg}, indent=2))
    for k in keys: print(f"[{name}/{model}] {k:8s} cov {agg[k]['coverage'][0]:.3f}±{agg[k]['coverage'][1]:.3f} width {agg[k]['width'][0]:.2f}", flush=True)

if __name__ == "__main__": main()
