"""
ResCP (Reservoir Conformal Prediction, Neglia et al., ICLR 2026) applied to the point forecasts.
Uses the Reservoir and the sampling rule of the official implementation
(third_party/reservoir-conformal-prediction). The target is observed one hour (6 steps) after the
forecast, so at step k only residuals r[j] with j <= k-6 are in the buffer and the query state is H[k-6].

Input   results/<name>/<model>/seed<k>/pred.csv  (ts, split, y_T, y_RH, p_T, p_RH)
Output  results/<name>/<model>/seed<k>/rescp_<target>_<tag>.csv  (ts, y, p, lo, hi) + rescp_metrics.json
        results/<name>/<model>/rescp_summary_<tag>.json (mean and std over seeds)

Settings used in the paper
  reservoir 512 units, leak 0.8, spectral radius 0.95, input scaling 0.25, connectivity 0.2
  residuals standardised with calibration statistics -> reservoir -> L2-normalised states
  cosine similarity, softmax temperature T = 0.3, linear time decay, buffer 2000, 2000 resamples,
  beta grid of 100 (minimum-width interval), alpha = 0.1

Usage: python run_rescp.py --name kangwon --model lgbm --T 0.3 --window 2000
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE.parent / "third_party" / "reservoir-conformal-prediction"))
from reservoir_conformal_prediction.src.torch_reservoir_computing.reservoir import Reservoir  # noqa: E402

LAG = 6                     # delay between forecast and target observation (steps)
TARGETS = {"T": ("y_T", "p_T"), "RH": ("y_RH", "p_RH")}
RES_HP = dict(n_internal_units=512, spectral_radius=0.95, leak=0.8, connectivity=0.2,
              input_scaling=0.25, noise_level=0.0, circle=False, seed=0)


def reservoir_states(r_std, dev):
    res = Reservoir(**RES_HP).to(dev)
    x = torch.as_tensor(r_std, dtype=torch.float32, device=dev)[None, :, None]   # (1, N, 1)
    H = res.get_states(x, bidir=False)[0]                                          # (N, 512)
    H = H / torch.linalg.norm(H, dim=1, keepdim=True).clamp_min(1e-12)
    return H


def rescp(y, p, n_cal, alpha, window, T, n_q, dev, seed=0):
    """y, p: 1-D arrays, calibration followed by test. Returns lo, hi for the test part."""
    g = torch.Generator(device=dev).manual_seed(seed)
    r = y - p
    mu, sd = r[:n_cal].mean(), r[:n_cal].std() + 1e-8
    H = reservoir_states((r - mu) / sd, dev)
    r_t = torch.as_tensor(r, dtype=torch.float32, device=dev)
    N = len(y)
    lower_b = torch.linspace(1e-3, alpha - 1e-3, n_q, device=dev)
    upper_b = 1 - alpha + lower_b
    lo = np.full(N - n_cal, np.nan); hi = np.full(N - n_cal, np.nan)
    for k in range(n_cal, N):
        avail = k - LAG                       # last index whose residual is observed at step k
        j0 = max(LAG, avail - window + 1)     # state H[j-LAG] is defined for j >= LAG
        js = torch.arange(j0, avail + 1, device=dev)
        states = H[js - LAG]                  # (n, 512) dynamics preceding residual r[j]
        q = H[avail]                          # query state
        sim = torch.softmax((states @ q) / T, dim=0)
        w = torch.arange(1, len(js) + 1, device=dev, dtype=torch.float32)   # linear decay (older residuals weighted less)
        w = sim * (w / w.sum()); w = w / w.sum()
        samp = r_t[js][torch.multinomial(w, num_samples=window, replacement=True, generator=g)]
        ql = torch.quantile(samp, lower_b); qu = torch.quantile(samp, upper_b)
        i = torch.argmin(qu - ql)
        lo[k - n_cal] = p[k] + ql[i].item(); hi[k - n_cal] = p[k] + qu[i].item()
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--T", type=float, default=0.3)
    ap.add_argument("--n_q", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    root = Path(a.results).expanduser() / a.name / a.model
    tag = f"a{int(round((1 - a.alpha) * 100))}_w{a.window}_T{a.T:g}"
    summary = {}
    for sd in sorted(root.glob("seed*")):
        df = pd.read_csv(sd / "pred.csv", parse_dates=["ts"])
        n_cal = int((df["split"] == "calib").sum())
        assert (df["split"].values[:n_cal] == "calib").all()
        met = {}
        for tg, (yc, pc) in TARGETS.items():
            t0 = time.time()
            y, p = df[yc].to_numpy(), df[pc].to_numpy()
            lo, hi = rescp(y, p, n_cal, a.alpha, a.window, a.T, a.n_q, dev)
            te = df.iloc[n_cal:].reset_index(drop=True)
            out = pd.DataFrame({"ts": te["ts"], "y": te[yc], "p": te[pc], "lo": lo, "hi": hi})
            out.to_csv(sd / f"rescp_{tg}_{tag}.csv", index=False)
            cov = float(((out.y >= out.lo) & (out.y <= out.hi)).mean())
            met[tg] = {"coverage": cov, "width": float((out.hi - out.lo).mean()),
                       "width_median": float((out.hi - out.lo).median()), "n": len(out), "sec": round(time.time() - t0, 1)}
            print(f"[{a.name}/{a.model}/{sd.name}] {tg}: coverage={cov:.3f} width={met[tg]['width']:.3f} ({met[tg]['sec']}s)", flush=True)
        mp = sd / "rescp_metrics.json"
        allm = json.loads(mp.read_text()) if mp.exists() else {}
        allm[tag] = met
        mp.write_text(json.dumps(allm, indent=2))
        summary[sd.name] = met
    # mean over seeds
    agg = {tg: {k: [float(np.mean([summary[s][tg][k] for s in summary])), float(np.std([summary[s][tg][k] for s in summary]))]
                for k in ["coverage", "width"]} for tg in TARGETS}
    (root / f"rescp_summary_{tag}.json").write_text(json.dumps({"seeds": summary, "mean_std": agg}, indent=2))
    for tg in TARGETS:
        print(f"[{a.name}/{a.model}] {tg}: coverage {agg[tg]['coverage'][0]:.3f}±{agg[tg]['coverage'][1]:.3f}  "
              f"width {agg[tg]['width'][0]:.3f}±{agg[tg]['width'][1]:.3f}")


if __name__ == "__main__":
    main()
