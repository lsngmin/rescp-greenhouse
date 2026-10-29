"""
Shared utilities for the forecasting models.
- load samples (npz written by make_samples.py)
- split the 73-day training period into 66 training days / 7 validation days (validation = last 7 days, from 00:00)
- standardisation statistics (66 training days only)
- common prediction format: results/<dataset>/<model>/seed<k>/pred.csv
    columns ts, split (calib|test), y_T, y_RH, p_T, p_RH
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

INP = ["T_in", "RH_in", "T_out", "RH_out", "WS_out", "Light_out", "Vent", "hour_sin", "hour_cos"]
TGT = ["T_in_max1h", "RH_in_max1h"]
VAL_DAYS = 7
SEEDS = [0, 1, 2, 3, 4]


class Samples:
    def __init__(self, root, name):
        z = np.load(Path(root).expanduser() / "samples" / f"{name}_samples.npz")
        self.name = name
        X, y, ts = z["X_train"], z["y_train"], pd.to_datetime(z["ts_train"])
        # validation = last 7 days of the training period (from 00:00 six days before the last day)
        val_start = ts.max().normalize() - pd.Timedelta(days=VAL_DAYS - 1)
        m = ts >= val_start
        self.X_train, self.y_train, self.ts_train = X[~m], y[~m], ts[~m]
        self.X_val, self.y_val, self.ts_val = X[m], y[m], ts[m]
        self.X_calib, self.y_calib, self.ts_calib = z["X_calib"], z["y_calib"], pd.to_datetime(z["ts_calib"])
        self.X_test, self.y_test, self.ts_test = z["X_test"], z["y_test"], pd.to_datetime(z["ts_test"])
        # standardisation statistics from the 66 training days only
        flat = self.X_train.reshape(-1, self.X_train.shape[-1])
        self.x_mean, self.x_std = flat.mean(0), flat.std(0) + 1e-8
        self.y_mean, self.y_std = self.y_train.mean(0), self.y_train.std(0) + 1e-8

    def summary(self):
        f = lambda t: f"{t.min():%Y-%m-%d %H:%M} ~ {t.max():%Y-%m-%d %H:%M}"
        return {
            "train": [len(self.ts_train), f(self.ts_train)],
            "val": [len(self.ts_val), f(self.ts_val)],
            "calib": [len(self.ts_calib), f(self.ts_calib)],
            "test": [len(self.ts_test), f(self.ts_test)],
        }

    def norm_x(self, X):
        return (X - self.x_mean) / self.x_std

    def norm_y(self, y):
        return (y - self.y_mean) / self.y_std

    def denorm_y(self, y):
        return y * self.y_std + self.y_mean


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def save_pred(out_dir, S, p_calib, p_test, extra=None):
    """Save calibration/test predictions in the common format and return RMSEs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, ts, y, p in [("calib", S.ts_calib, S.y_calib, p_calib), ("test", S.ts_test, S.y_test, p_test)]:
        df = pd.DataFrame({"ts": ts, "split": split, "y_T": y[:, 0], "y_RH": y[:, 1], "p_T": p[:, 0], "p_RH": p[:, 1]})
        rows.append(df)
    pd.concat(rows).to_csv(out_dir / "pred.csv", index=False)
    metrics = {
        "rmse_calib_T": rmse(S.y_calib[:, 0], p_calib[:, 0]), "rmse_calib_RH": rmse(S.y_calib[:, 1], p_calib[:, 1]),
        "rmse_test_T": rmse(S.y_test[:, 0], p_test[:, 0]), "rmse_test_RH": rmse(S.y_test[:, 1], p_test[:, 1]),
    }
    if extra:
        metrics.update(extra)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
