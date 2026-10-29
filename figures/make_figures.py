"""All figures and the baseline table of the paper from results/, results_sim/ and data/.

  python figures/make_figures.py [--results results] [--results-sim results_sim/closed_2011] [--data data] [--out figs] [--only figA figB1 ...]

Outputs (PDF + PNG, 300 dpi) in --out
  figV_datasets_overview_1col   daily indoor T/RH of both measured datasets with the split boundaries
  figA_rmse_vs_width            point-forecast RMSE vs mean ResCP interval width (28 conditions)
  figA2_rmse_by_period          RMSE on validation / calibration / test per forecaster
  figB1_weekly_coverage         7-day rolling ResCP coverage, measured data
  figB2_interval_example        three-day example of the interval band (Korea, LightGBM, temperature)
  figB3_baselines               coverage vs width of ResCP, sliding-window CP and ACI; also table_baselines_28conditions.csv
  figC_dumbbell                 closed loop: baseline vs interval control, coverage and width per forecaster
  figC2_closed_loop_rolling_coverage   7-day rolling coverage in the closed loop
  figD1_closed_loop_daily       daily share of time above 85 % RH and daily mean vent opening
  figD2_intervention_vs_coverage       share of time with the humidity set point adjusted vs humidity coverage, five groups
"""
import argparse, json, glob
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

M = ["lgbm", "gru", "xlstm", "moderntcn", "timemixer", "patchtst", "itransformer"]
NAME = {"lgbm": "LightGBM", "gru": "GRU", "xlstm": "xLSTM", "moderntcn": "ModernTCN", "timemixer": "TimeMixer", "patchtst": "PatchTST", "itransformer": "iTransformer"}
DS = {"kangwon": "Korea", "agc1_ref": "Netherlands"}
TAG = "a90_w2000_T0.3"
TEAL = "#0F766E"; AMBER = "#C2410C"; GRAY = "#6B7280"; INK = "#1F2A37"; LIGHT = "#CBD5E1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK, "axes.linewidth": 0.6, "pdf.fonttype": 42})
def weekly_axis(ax):  # locator/formatter objects must not be shared between axes
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=7)); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("saved", name)


def seed_metrics(root):
    return [json.load(open(f)) for f in glob.glob(str(root / "seed*/metrics.json"))]


def open_summary():
    rows = []
    for ds in DS:
        for m in M:
            d = json.load(open(R / ds / m / f"rescp_summary_{TAG}.json"))["mean_std"]
            js = seed_metrics(R / ds / m)
            for tg in ["T", "RH"]:
                rows.append(dict(ds=ds, model=m, var=tg, rmse=np.mean([j[f"rmse_test_{tg}"] for j in js]), cov=d[tg]["coverage"][0], width=d[tg]["width"][0]))
    return pd.DataFrame(rows)


def closed(mode, m):
    ms = seed_metrics(S / mode / m)
    return {k: np.mean([x[k] for x in ms]) for k in ["cov_T", "cov_RH", "width_T", "width_RH", "RH_over85_frac", "vent_mean", "int_RH_frac"]}


def traces(mode, m):
    return [pd.read_csv(f, parse_dates=["ts"]).set_index("ts") for f in sorted(glob.glob(str(S / mode / m / "seed*/trace.csv")))]


# ------------------------------------------------------------------ figV
def figV():
    sp = json.load(open(D / "splits.json"))
    VARS = [("T_in", "Indoor\ntemperature (°C)"), ("RH_in", "Indoor relative\nhumidity (%)")]
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.3), dpi=150, sharex="col")
    for j, (ds, name) in enumerate(DS.items()):
        df = pd.read_parquet(D / f"{ds}.parquet"); s = sp[ds]
        bounds = [pd.Timestamp(s["train"]["end"]) - pd.Timedelta(days=6), pd.Timestamp(s["calib"]["start"]), pd.Timestamp(s["test"]["start"])]
        starts = [pd.Timestamp(s["train"]["start"])] + bounds; ends = bounds + [pd.Timestamp(s["test"]["end"]) + pd.Timedelta(days=1)]
        for i, (v, lab) in enumerate(VARS):
            ax = axes[i, j]; d = df[v].resample("1D")
            ax.fill_between(d.mean().index, d.min(), d.max(), color=TEAL, alpha=0.18, lw=0, label="Daily min–max")
            ax.plot(d.mean().index, d.mean(), color=TEAL, lw=0.9, label="Daily mean")
            for b in bounds: ax.axvline(b, color=AMBER, lw=0.6, ls="--")
            ax.axvspan(starts[3], ends[3], color=AMBER, alpha=0.05, lw=0)
            if i == 0:
                ax.set_title(name, loc="left", pad=10)
                for st_, en, lb in zip([starts[0], starts[3]], [ends[0], ends[3]], ["Train", "Test"]):
                    ax.text(st_ + (en - st_) / 2, 1.02, lb, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6, color=AMBER)
            if j == 0: ax.set_ylabel(lab)
            ax.grid(axis="y", color="#EEF2F7", lw=0.6)
            sd_tr = df[v][:s["train"]["end"]].std(); sd_te = df[v][s["test"]["start"]:].std()
            ax.text(0.01, 0.96, f"SD {sd_tr:.1f} → {sd_te:.1f}", transform=ax.transAxes, ha="left", va="top", fontsize=5.5, color=GRAY, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator()); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b")); ax.tick_params(labelsize=6)
    h, l = axes[1, 0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc="upper center", ncol=2, fontsize=6, bbox_to_anchor=(0.55, 1.0), handlelength=1.5, columnspacing=1.0)
    fig.align_ylabels(axes[:, 0]); fig.tight_layout(h_pad=0.4, w_pad=0.6, rect=(0, 0, 1, 0.95)); save(fig, "figV_datasets_overview_1col")


# ------------------------------------------------------------------ figA, figA2
def figA():
    OS = open_summary()
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), dpi=150)
    for ax, tg, unit in zip(axes, ["T", "RH"], ["°C", "%p"]):
        for ds, mk in zip(DS, ["o", "s"]):
            d = OS[(OS["var"] == tg) & (OS.ds == ds)]
            ax.scatter(d.rmse, d.width, marker=mk, s=18, facecolor="white", edgecolor=TEAL, linewidth=1, label=DS[ds], zorder=3)
        d = OS[OS["var"] == tg]; k = np.polyfit(d.rmse, d.width, 1); xs = np.linspace(d.rmse.min(), d.rmse.max(), 10)
        ax.plot(xs, np.polyval(k, xs), color=LIGHT, lw=1, zorder=1)
        ax.set_xlabel(f"RMSE ({unit})"); ax.set_ylabel(f"Mean interval width ({unit})"); ax.set_title("Temperature" if tg == "T" else "Relative humidity", loc="left")
        ax.grid(axis="both", color="#EEF2F7", lw=0.6)
    axes[0].legend(frameon=False, loc="upper left", handletextpad=0.2)
    fig.tight_layout(w_pad=1.0); save(fig, "figA_rmse_vs_width")


def figA2():
    C = ["#A7D3CC", "#4FA39A", "#0F766E"]  # validation, calibration, test
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.9), dpi=150, sharex=True)
    x = np.arange(len(M))
    for j, ds in enumerate(DS):
        for i, (tg, unit) in enumerate([("T", "°C"), ("RH", "%p")]):
            ax = axes[i, j]; vals = {k: [] for k in ["val", "calib", "test"]}
            for m in M:
                js = seed_metrics(R / ds / m)
                vals["val"].append(np.mean([q[f"val_rmse_{tg}"] for q in js])); vals["calib"].append(np.mean([q[f"rmse_calib_{tg}"] for q in js])); vals["test"].append(np.mean([q[f"rmse_test_{tg}"] for q in js]))
            for xi in x: ax.plot([xi, xi], [min(vals["val"][xi], vals["test"][xi]), max(vals["val"][xi], vals["test"][xi])], color="#E2E8F0", lw=2, zorder=1)
            for k, c, lab in zip(["val", "calib", "test"], C, ["Validation (7 d)", "Calibration (14 d)", "Test (28 d)"]):
                ax.scatter(x, vals[k], s=14, color=c, zorder=3, label=lab, edgecolor="white", linewidth=0.4)
            ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in M], rotation=50, ha="right")
            if j == 0: ax.set_ylabel(("Temperature" if tg == "T" else "Rel. humidity") + f"\nRMSE ({unit})")
            if i == 0: ax.set_title(DS[ds], loc="left")
            ax.grid(axis="y", color="#EEF2F7", lw=0.6)
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.55, 1.0), handletextpad=0.1, columnspacing=1.0)
    fig.tight_layout(h_pad=0.5, w_pad=0.8, rect=(0, 0, 1, 0.95)); save(fig, "figA2_rmse_by_period")


# ------------------------------------------------------------------ figB1, figB2, figB3
def figB1():
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.2), dpi=150, sharex="col")
    for j, ds in enumerate(DS):
        for i, tg in enumerate(["T", "RH"]):
            ax = axes[i, j]
            for m in M:
                dfs = [pd.read_csv(f, parse_dates=["ts"]) for f in sorted(glob.glob(str(R / ds / m / f"seed*/rescp_{tg}_{TAG}.csv")))]
                hit = np.mean([((d.y >= d.lo) & (d.y <= d.hi)).astype(float).values for d in dfs], axis=0)
                s = pd.Series(hit, index=dfs[0].ts).resample("1D").mean().rolling(7, center=True, min_periods=4).mean()
                ax.plot(s.index, s.values, color=TEAL, lw=0.8, alpha=0.6)
            ax.axhline(0.90, color=AMBER, ls="--", lw=0.8); ax.set_ylim(0.80, 1.0)
            if j == 0: ax.set_ylabel(("Temperature" if tg == "T" else "Rel. humidity") + "\n7-day rolling coverage")
            if i == 0: ax.set_title(DS[ds], loc="left")
            ax.grid(axis="y", color="#EEF2F7", lw=0.6); weekly_axis(ax)
    axes[0, 1].text(0.99, 0.93, "target 0.90", transform=axes[0, 1].transAxes, ha="right", va="top", color=AMBER, fontsize=7)
    fig.tight_layout(h_pad=0.6, w_pad=1.0); save(fig, "figB1_weekly_coverage")


def figB2():
    d = pd.read_csv(R / "kangwon/lgbm/seed0" / f"rescp_T_{TAG}.csv", parse_dates=["ts"]).set_index("ts")
    seg = d["2023-11-05":"2023-11-07 23:50"]
    fig, ax = plt.subplots(figsize=(3.5, 1.9), dpi=150)
    ax.fill_between(seg.index, seg.lo, seg.hi, color=TEAL, alpha=0.18, lw=0, label="90 % prediction interval")
    ax.plot(seg.index, seg.p, color=TEAL, lw=0.8, label="Point forecast")
    ax.plot(seg.index, seg.y, color=INK, lw=0.8, label="Observed 1-h maximum")
    miss = seg[(seg.y < seg.lo) | (seg.y > seg.hi)]
    ax.scatter(miss.index, miss.y, s=4, color=AMBER, zorder=4, label="Outside interval")
    ax.set_ylabel("Indoor air temperature (°C)"); ax.xaxis.set_major_locator(matplotlib.dates.DayLocator()); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=[12])); ax.grid(axis="y", color="#EEF2F7", lw=0.6)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6, handlelength=1.5)
    cov = ((seg.y >= seg.lo) & (seg.y <= seg.hi)).mean(); ax.text(0.99, 0.03, f"coverage in window {cov:.3f}", transform=ax.transAxes, ha="right", fontsize=6.5, color=GRAY)
    fig.tight_layout(); save(fig, "figB2_interval_example")


def figB3():
    rows = []
    for ds in DS:
        for m in M:
            r = json.load(open(R / ds / m / f"rescp_summary_{TAG}.json"))["mean_std"]; b = json.load(open(R / ds / m / "base_summary.json"))["mean_std"]
            for tg in ["T", "RH"]:
                rows.append((ds, m, tg, "ResCP", r[tg]["coverage"][0], r[tg]["width"][0]))
                rows.append((ds, m, tg, "SWCP", b[f"swcp_{tg}"]["coverage"][0], b[f"swcp_{tg}"]["width"][0]))
                rows.append((ds, m, tg, "ACI", b[f"aci_{tg}"]["coverage"][0], b[f"aci_{tg}"]["width"][0]))
    tab = pd.DataFrame(rows, columns=["dataset", "model", "variable", "method", "coverage", "width"])
    tab["dataset"] = tab.dataset.map(DS); tab["model"] = tab.model.map(NAME)
    tab.to_csv(OUT / "table_baselines_28conditions.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.4), dpi=150)
    MK = {"ResCP": ("o", TEAL, "ResCP"), "SWCP": ("s", GRAY, "Sliding-window CP"), "ACI": ("^", AMBER, "ACI")}
    for i, tg in enumerate(["T", "RH"]):
        for j, ds in enumerate(DS):
            ax = axes[i, j]
            for meth, (mk, col, lab) in MK.items():
                d = [x for x in rows if x[0] == ds and x[2] == tg and x[3] == meth]
                ax.scatter([x[5] for x in d], [x[4] for x in d], marker=mk, s=14, facecolor="white", edgecolor=col, linewidth=1, label=lab, zorder=3)
            ax.axhline(0.90, color=INK, ls="--", lw=0.6)
            if i == 0: ax.set_title(DS[ds], loc="left")
            if j == 0: ax.set_ylabel(("Temperature" if tg == "T" else "Rel. humidity") + "\ncoverage")
            ax.set_xlabel("Mean width (°C)" if tg == "T" else "Mean width (%p)", fontsize=7)
            ax.grid(color="#EEF2F7", lw=0.6); ax.set_ylim(0.84, 0.94); ax.tick_params(labelsize=6)
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc="upper center", ncol=3, fontsize=6, bbox_to_anchor=(0.55, 1.0), handletextpad=0.1, columnspacing=0.8)
    fig.tight_layout(h_pad=0.6, w_pad=0.6, rect=(0, 0, 1, 0.95)); save(fig, "figB3_baselines")


# ------------------------------------------------------------------ figC, figC2
def figC():
    CB = {m: closed("baseline", m) for m in M}; CD = {m: closed("D2", m) for m in M}
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.2), dpi=150, sharey=True)
    y = np.arange(len(M))[::-1]
    for ax, key, title in zip(axes, ["cov_T", "cov_RH", "width_T", "width_RH"], ["Temperature coverage", "Rel. humidity coverage", "Temperature width (°C)", "Rel. humidity width (%p)"]):
        b = [CB[m][key] for m in M]; d2 = [CD[m][key] for m in M]
        for yi, bb, dd in zip(y, b, d2): ax.plot([bb, dd], [yi, yi], color=LIGHT, lw=2, zorder=1)
        ax.scatter(b, y, s=18, color=GRAY, zorder=3, label="Baseline"); ax.scatter(d2, y, s=18, color=AMBER, zorder=3, label="Interval control")
        if "cov" in key: ax.axvline(0.90, color=AMBER, ls="--", lw=0.7); ax.set_xlim(0.87, 0.925)
        ax.set_title(title, loc="left"); ax.grid(axis="x", color="#EEF2F7", lw=0.6)
    axes[0].set_yticks(y); axes[0].set_yticklabels([NAME[m] for m in M])
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2, fontsize=7)
    fig.tight_layout(w_pad=0.8); save(fig, "figC_dumbbell")


def figC2():
    def roll(mode, m, tg):
        dfs = traces(mode, m); y, lo, hi = f"y{tg}", f"lo{tg}", f"hi{tg}"
        hit = np.mean([((d[y] >= d[lo]) & (d[y] <= d[hi])).astype(float).values for d in dfs], axis=0)
        return pd.Series(hit, index=dfs[0].index).resample("1D").mean().rolling(7, center=True, min_periods=4).mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3), dpi=150, sharey=True)
    for ax, tg, title in zip(axes, ["T", "RH"], ["Temperature", "Relative humidity"]):
        for m in M:
            b = roll("baseline", m, tg); d = roll("D2", m, tg)
            ax.plot(b.index, b.values, color=GRAY, lw=0.8, alpha=0.6); ax.plot(d.index, d.values, color=AMBER, lw=0.8, alpha=0.7)
        ax.axhline(0.90, color=INK, ls="--", lw=0.7); ax.set_ylim(0.75, 1.0); ax.set_title(title, loc="left"); ax.grid(axis="y", color="#EEF2F7", lw=0.6)
        weekly_axis(ax)
    axes[0].set_ylabel("7-day rolling coverage")
    h = [Line2D([], [], color=GRAY, lw=1.2), Line2D([], [], color=AMBER, lw=1.2), Line2D([], [], color=INK, ls="--", lw=0.8)]
    axes[1].legend(h, ["Baseline (7 forecasters)", "Interval control (7 forecasters)", "Target 0.90"], frameon=False, loc="lower right", ncol=1, fontsize=6.5)
    fig.tight_layout(w_pad=1.0); save(fig, "figC2_closed_loop_rolling_coverage")


# ------------------------------------------------------------------ figD1, figD2
def figD1():
    def daily(mode, m):
        ts = traces(mode, m)
        over = np.mean([(t.RH_in > 85).astype(float).resample("1D").mean().values for t in ts], axis=0)
        vent = np.mean([t.Vent.resample("1D").mean().values for t in ts], axis=0)
        return ts[0].RH_in.resample("1D").mean().index, over * 100, vent
    idx, ob, vb = daily("baseline", "lgbm")
    Dd = [daily("D2", m) for m in M]; od = np.array([d[1] for d in Dd]); vd = np.array([d[2] for d in Dd])
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 3.0), dpi=150, sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    ax = axes[0]
    ax.fill_between(idx, od.min(0), od.max(0), color=AMBER, alpha=0.18, lw=0, label="Interval control, range over 7 forecasters")
    ax.plot(idx, od.mean(0), color=AMBER, lw=1.0, label="Interval control, mean"); ax.plot(idx, ob, color=GRAY, lw=1.0, label="Baseline")
    ax.set_ylabel("Share of day above\n85 % RH (%)"); ax.set_ylim(0, 100); ax.grid(axis="y", color="#EEF2F7", lw=0.6)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    ax = axes[1]
    ax.fill_between(idx, vd.min(0), vd.max(0), color=AMBER, alpha=0.18, lw=0); ax.plot(idx, vd.mean(0), color=AMBER, lw=1.0); ax.plot(idx, vb, color=GRAY, lw=1.0)
    ax.set_ylabel("Daily mean vent\nopening (%)"); ax.grid(axis="y", color="#EEF2F7", lw=0.6); weekly_axis(ax)
    fig.align_ylabels(axes); fig.tight_layout(h_pad=0.4); save(fig, "figD1_closed_loop_daily")


def figD2():
    def agg(mode):
        return np.array([(closed(mode, m)["int_RH_frac"] * 100, closed(mode, m)["cov_RH"]) for m in M])
    G = [("baseline", "Baseline", "o", GRAY), ("P3", "Point, δ = 1.5 °C", "v", "#9CA3AF"), ("P2", "Point, δ = 1.0 °C", "^", "#6B7280"), ("D2", "Interval control", "o", AMBER), ("P", "Point, δ = 0", "s", "#1F2A37")]
    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=150)
    for mode, lab, mk, col in G:
        a = agg(mode); ax.scatter(a[:, 0], a[:, 1], marker=mk, s=18, facecolor="white" if mode != "D2" else AMBER, edgecolor=col, linewidth=1, label=lab, zorder=3 if mode != "D2" else 4)
    ax.axhline(0.90, color=INK, ls="--", lw=0.6)
    ax.set_xlabel("Share of time with humidity setpoint adjusted (%)"); ax.set_ylabel("Rel. humidity coverage")
    ax.set_ylim(0.86, 0.92); ax.grid(color="#EEF2F7", lw=0.6)
    ax.legend(frameon=False, loc="lower left", ncol=2, handletextpad=0.2, columnspacing=0.8, fontsize=6.5)
    fig.tight_layout(); save(fig, "figD2_intervention_vs_coverage")


FIGS = {"figV": figV, "figA": figA, "figA2": figA2, "figB1": figB1, "figB2": figB2, "figB3": figB3, "figC": figC, "figC2": figC2, "figD1": figD1, "figD2": figD2}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--results-sim", default="results_sim/closed_2011"); ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="figs"); ap.add_argument("--only", nargs="*", choices=list(FIGS))
    a = ap.parse_args()
    R, S, D, OUT = Path(a.results), Path(a.results_sim), Path(a.data), Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    for k in (a.only or FIGS): FIGS[k]()
