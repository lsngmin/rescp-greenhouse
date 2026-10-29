"""One closed-loop run in GreenLight-Gym (conda env glgym).

Groups (--mode)
  baseline : rule-based controller only. The forecaster and ResCP run during the test period and are recorded,
             nothing is changed (control group).
  D2       : interval-based control (the "interval control" group of the paper). Every 10 min, for the next step:
             temperature  lower bound > temperature vent threshold -> heat_deadzone - 2*DT; only upper bound above -> - DT
                          no action when T_out >= T_in (ventilation cannot cool)
             humidity     lower bound > humidity vent threshold -> rh_max - 2*DRH; only upper bound above -> - DRH
                          no action when outdoor vapour pressure >= indoor (ventilation cannot dehumidify)
             cold rule    temperature upper bound < heating set point -> no humidity action
                          temperature lower bound < heating set point <= upper bound -> humidity action one level lower
  P, P2, P3: point-forecast control with the same thresholds (control groups). One level when the point forecast
             exceeds the threshold; no humidity action when p_T - delta < heating set point, delta = 0 / 1.0 / 1.5 degC.
  DT = 1 degC, DRH = 5 %-points (width of the humidity proportional band of the rule-based controller)

ResCP settings are those of the measured-data experiments (T = 0.3, buffer 2000, 6-step delay, reservoir as in the
original paper). The buffer is initialised with the calibration residuals of the offline run.

Usage:  python simulation/run_closed_loop.py --model lgbm --seed 0 --mode D2 --year 2011
Output: results_sim/closed_<year>/<mode>/<model>/seed<k>/{trace.csv, metrics.json}
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation")); sys.path.insert(0, str(ROOT / "forecasting")); sys.path.insert(0, str(ROOT / "conformal"))
from sim_common import make, ctx_of, record, RuleBasedController, satVp   # noqa: E402
from common import Samples, INP                                            # noqa: E402
import run_rescp as R                                                      # noqa: E402

W, LAG = 6, R.LAG
ALPHA, TEMP, WINDOW, NQ = 0.1, 0.3, 2000, 100
DT, DRH = 1.0, 5.0


# ---------------------------------------------------------------- forecaster
def load_predictor(pred_dir, model, seed, S, dev):
    sd = pred_dir / model / f"seed{seed}"
    if model == "lgbm":
        import lightgbm as lgb
        ms = [lgb.Booster(model_file=str(sd / f"model_{j}.txt")) for j in range(2)]
        return lambda X: np.array([m.predict(X.reshape(1, -1))[0] for m in ms])
    import optuna, run_dl
    best = json.loads((pred_dir / model / "tuning.json").read_text())["best_params"]
    hp = run_dl.suggest(optuna.trial.FixedTrial(best), model)
    net = run_dl.MODELS[model](hp).to(dev)
    net.load_state_dict(torch.load(sd / "model.pt", map_location=dev)); net.eval()

    @torch.no_grad()
    def f(X):
        x = torch.as_tensor(S.norm_x(X[None]), dtype=torch.float32, device=dev)
        return S.denorm_y(net(x).float().cpu().numpy())[0]
    return f


# ---------------------------------------------------------------- online ResCP (same rule as run_rescp.rescp)
class OnlineResCP:
    def __init__(self, r_cal, r_stat, dev, seed=0):
        """r_cal: residuals for the initial buffer; r_stat: residuals for the standardisation statistics
        (whole 14-day calibration period, as in run_rescp)."""
        self.dev = dev
        self.mu, self.sd = float(r_stat.mean()), float(r_stat.std() + 1e-8)
        self.res = R.Reservoir(**R.RES_HP).to(dev)
        self.state = None
        self.H, self.r = [], []                      # H[j] = L2-normalised state after feeding r[0..j]
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.lb = torch.linspace(1e-3, ALPHA - 1e-3, NQ, device=dev); self.ub = 1 - ALPHA + self.lb
        for v in r_cal:
            self.push(float(v))

    def push(self, r):
        x = torch.tensor([[[(r - self.mu) / self.sd]]], dtype=torch.float32, device=self.dev)
        h = self.res.get_states(x, bidir=False, initial_state=self.state)[0]      # (1, 512)
        self.state = h[-1:].clone()
        self.H.append((h[-1] / torch.linalg.norm(h[-1]).clamp_min(1e-12))); self.r.append(r)

    def interval(self, p):
        """Interval from the residuals observed so far r[0..avail]: query state H[avail], buffer (H[j-LAG], r[j])."""
        avail = len(self.r) - 1
        j0 = max(LAG, avail - WINDOW + 1)
        js = torch.arange(j0, avail + 1, device=self.dev)
        states = torch.stack(self.H[j0 - LAG: avail - LAG + 1])
        q = self.H[avail]
        sim = torch.softmax((states @ q) / TEMP, dim=0)
        w = torch.arange(1, len(js) + 1, device=self.dev, dtype=torch.float32)
        w = sim * (w / w.sum()); w = w / w.sum()
        rr = torch.tensor(self.r[j0: avail + 1], dtype=torch.float32, device=self.dev)
        samp = rr[torch.multinomial(w, num_samples=WINDOW, replacement=True, generator=self.g)]
        ql, qu = torch.quantile(samp, self.lb), torch.quantile(samp, self.ub)
        i = torch.argmin(qu - ql)
        return p + ql[i].item(), p + qu[i].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["baseline", "D2", "P", "P2", "P3"], required=True); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--year", type=int, default=2011)
    ap.add_argument("--data", default="data_sim"); ap.add_argument("--results", default="results_sim")
    a = ap.parse_args()
    name = f"sim_{a.year}"
    data = Path(a.data).resolve(); results = Path(a.results).resolve()      # resolve before make() changes the cwd
    pred_dir = results / "pred" / name
    out_root = results / f"closed_{a.year}"
    TEST_START = pd.Timestamp(f"{a.year}-11-09 00:00")
    ONLINE_START = TEST_START - pd.Timedelta(minutes=60)   # online forecasts start 6 steps before the test period; actions from TEST_START
    END = pd.Timestamp(f"{a.year}-12-07")
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    S = Samples(data, name)
    pred = load_predictor(pred_dir, a.model, a.seed, S, dev)

    # calibration residuals from the offline prediction file (open-loop run)
    pc = pd.read_csv(pred_dir / a.model / f"seed{a.seed}" / "pred.csv", parse_dates=["ts"])
    cal = pc[pc.split == "calib"]
    # the targets of the last 6 calibration samples are observed in the first hour of the test period, so they are
    # left out of the initial buffer and refilled by the online forecasts (from ONLINE_START)
    assert cal.ts.iloc[-1] == ONLINE_START + pd.Timedelta(minutes=10 * (LAG - 1))
    rT, rR = (cal.y_T - cal.p_T).to_numpy(), (cal.y_RH - cal.p_RH).to_numpy()
    cps = {"T": OnlineResCP(rT[:-LAG], rT, dev), "RH": OnlineResCP(rR[:-LAG], rR, dev)}

    env, rb, scen = make(year=a.year); ctl = RuleBasedController(**rb)
    base_dz, base_rh = ctl.heat_deadzone, ctl.rh_max
    env.reset(seed=0, options={"scenario": scen})
    ts0 = pd.Timestamp(f"{a.year}-01-01") + pd.Timedelta(days=scen["start_day"] - 1)
    rows, preds = [], []        # preds[i] = (ts, pT, loT, hiT, pRH, loRH, hiRH, levelT, levelRH)
    tot = {"heat": 0., "elec": 0., "co2": 0., "rev": 0., "profit": 0.}
    t0 = time.time(); u_prev = None; vent_move = 0.
    while True:
        # --- forecast, interval and action decision at the current time t (last recorded step)
        lvT = lvR = 0
        if rows and rows[-1]["ts"] >= ONLINE_START and len(rows) >= W:
            X = np.array([[r[c] for c in INP] for r in rows[-W:]], dtype=np.float32)
            # delayed residual update: the target of the forecast issued 6 steps ago, max(T_in[t-5..t]), is now observed
            if len(preds) >= LAG:
                k = len(preds) - LAG
                assert preds[k]["ts"] == rows[-LAG - 1]["ts"], (preds[k]["ts"], rows[-LAG - 1]["ts"])
                yT = max(r["T_in"] for r in rows[-LAG:]); yR = max(r["RH_in"] for r in rows[-LAG:])
                cps["T"].push(yT - preds[k]["pT"]); cps["RH"].push(yR - preds[k]["pRH"])
            p = pred(X); cur = rows[-1]
            if cur["ts"] >= TEST_START:     # intervals from the test period on (same random sequence as the offline run_rescp)
                loT, hiT = cps["T"].interval(float(p[0])); loR, hiR = cps["RH"].interval(float(p[1]))
            else:
                loT = hiT = loR = hiR = float("nan")
            if a.mode == "D2" and cur["ts"] >= TEST_START:
                vp_in = cur["RH_in"] / 100 * satVp(cur["T_in"]); vp_out = cur["RH_out"] / 100 * satVp(cur["T_out"])
                if cur["T_out"] < cur["T_in"]:
                    lvT = 2 if loT > cur["T_vent_sp"] else (1 if hiT > cur["T_vent_sp"] else 0)
                if vp_out < vp_in:
                    lvR = 2 if loR > base_rh else (1 if hiR > base_rh else 0)
                if lvR > 0:
                    heat_sp = cur["T_vent_sp"] - base_dz            # lower reference = heating set point
                    if hiT < heat_sp:
                        lvR = 0                                     # certainly getting cold
                    elif loT < heat_sp:
                        lvR -= 1                                    # may get cold
            if a.mode in ("P", "P2", "P3") and cur["ts"] >= TEST_START:   # point-forecast control: one level when the point forecast exceeds the threshold
                vp_in = cur["RH_in"] / 100 * satVp(cur["T_in"]); vp_out = cur["RH_out"] / 100 * satVp(cur["T_out"])
                if cur["T_out"] < cur["T_in"]:
                    lvT = 1 if p[0] > cur["T_vent_sp"] else 0
                if vp_out < vp_in:
                    lvR = 1 if p[1] > base_rh else 0
                marg = {"P": 0.0, "P2": 1.0, "P3": 1.5}[a.mode]   # cold margin delta (degC): point forecast - delta replaces the interval lower bound
                if lvR > 0 and p[0] - marg < cur["T_vent_sp"] - base_dz:
                    lvR = 0
            preds.append({"ts": cur["ts"], "pT": float(p[0]), "loT": loT, "hiT": hiT,
                          "pRH": float(p[1]), "loRH": loR, "hiRH": hiR, "lvT": lvT, "lvRH": lvR})
        ctl.heat_deadzone = base_dz - DT * lvT
        ctl.rh_max = base_rh - DRH * lvR
        u = ctl.predict(ctx_of(env)); _, rwd, term, trunc, info = env.step(u.astype(np.float32))
        ctl.heat_deadzone, ctl.rh_max = base_dz, base_rh          # record the unmodified thresholds
        rec = record(env, ctl); rec["ts"] = ts0 + pd.Timedelta(seconds=env.dt * env.timestep); rows.append(rec)
        if rec["ts"] >= TEST_START:
            for k2, kk in [("heat", "heat_cost"), ("elec", "elec_cost"), ("co2", "co2_cost"), ("rev", "revenue"), ("profit", "profit")]:
                tot[k2] += float(info[kk])
            if u_prev is not None: vent_move += abs(float(env.u[3]) - u_prev)
            u_prev = float(env.u[3])
        if term or trunc: break

    tr = pd.DataFrame(rows).set_index("ts")
    pr = pd.DataFrame(preds).set_index("ts")
    # realised 1-h maxima along this run's own trajectory
    for v, c in [("T_in", "yT"), ("RH_in", "yRH")]:
        tr[c] = pd.concat([tr[v].shift(-k) for k in range(1, 7)], axis=1).max(axis=1, skipna=False)
    pr = pr.join(tr[["yT", "yRH", "T_in", "RH_in", "T_vent_sp", "Vent"]])
    pr = pr[(pr.index >= TEST_START) & (pr.index < END) & pr.yT.notna()]
    out = out_root / a.mode / a.model / f"seed{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    pr.to_csv(out / "trace.csv")
    te = tr[tr.index >= TEST_START]
    met = {
        "cov_T": float(((pr.yT >= pr.loT) & (pr.yT <= pr.hiT)).mean()), "width_T": float((pr.hiT - pr.loT).mean()),
        "cov_RH": float(((pr.yRH >= pr.loRH) & (pr.yRH <= pr.hiRH)).mean()), "width_RH": float((pr.hiRH - pr.loRH).mean()),
        "rmse_T": float(np.sqrt(((pr.yT - pr.pT) ** 2).mean())), "rmse_RH": float(np.sqrt(((pr.yRH - pr.pRH) ** 2).mean())),
        "n": int(len(pr)),
        "int_T_frac": float((pr.lvT > 0).mean()), "int_T_lv2_frac": float((pr.lvT == 2).mean()),
        "int_RH_frac": float((pr.lvRH > 0).mean()), "int_RH_lv2_frac": float((pr.lvRH == 2).mean()),
        "T_over_ventsp_frac": float((te.T_in > te.T_vent_sp).mean()), "RH_over85_frac": float((te.RH_in > 85).mean()),
        "T_in_mean": float(te.T_in.mean()), "RH_in_mean": float(te.RH_in.mean()), "vent_mean": float(te.Vent.mean()),
        "vent_move": vent_move, **{f"eur_{k}": v for k, v in tot.items()}, "sec": round(time.time() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps(met, indent=2))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()}))


if __name__ == "__main__":
    main()
