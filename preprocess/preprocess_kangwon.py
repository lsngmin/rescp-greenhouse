"""Korea (Gangwon sweet pepper) dataset -> 10-min table of the nine input variables.

Inputs
  --raw   HTVT_validation_10min.csv from Mendeley Data (doi:10.17632/mwg293k2r9.2)
  --kma   KMA ASOS station 211 1-min observations (CSV, cp949) from data.kma.go.kr
Output
  --out   parquet with columns T_in, RH_in, T_out, RH_out, WS_out, Light_out, Vent, hour_sin, hour_cos

Processing
  * timestamps: 'MM/DD HH:MM:SS' labels the end of each 10-min interval; 24:00:00 -> next day 00:00
  * rows flagged gap_excluded=1 and the undocumented interpolated block 2023-07-10 01:00-14:30 -> NaN
  * outdoor temperature, humidity and wind speed: KMA ASOS 1-min record sampled at each 10-min mark
  * irradiance from the on-site pyranometer; vent = leeward vent opening from the controller log
"""
import argparse, numpy as np, pandas as pd

RAMP = ("2023-07-10 01:00", "2023-07-10 14:30")

def hour_features(ts):
    minute = ts.hour * 60 + ts.minute
    return np.sin(2 * np.pi * minute / 1440), np.cos(2 * np.pi * minute / 1440)

def main(raw, kma, out):
    df = pd.read_csv(raw)
    s = df["Date/Time"].str.replace(r"\s+", " ", regex=True).str.strip()
    is24 = s.str.endswith("24:00:00")
    ts = pd.to_datetime("2023/" + s.where(~is24, s.str.replace("24:00:00", "00:00:00")), format="%Y/%m/%d %H:%M:%S")
    ts = ts + pd.to_timedelta(is24.astype(int), unit="D")
    df.index = ts; df.index.name = "ts"
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()

    bad = (df["gap_excluded [-]"] == 1) | ((df.index >= pd.Timestamp(RAMP[0])) & (df.index <= pd.Timestamp(RAMP[1])))
    T, RH = df["obs_Ta_cropzone [C]"].copy(), df["obs_RH_cropzone [%]"].copy()
    L, V = df["obs_Rad_out [W/m2]"].copy(), df["obs_VP_Lee [%]"].copy()
    for x in (T, RH, L, V): x[bad] = np.nan

    k = pd.read_csv(kma, encoding="cp949")
    k.columns = ["stn", "ts", "TA", "RN", "WD", "WS", "PA", "PS", "HM", "SI", "SS"]
    k["ts"] = pd.to_datetime(k["ts"]); k = k.set_index("ts").sort_index()
    hs, hc = hour_features(df.index)
    res = pd.DataFrame({"T_in": T, "RH_in": RH,
                        "T_out": k["TA"].reindex(df.index), "RH_out": k["HM"].reindex(df.index), "WS_out": k["WS"].reindex(df.index),
                        "Light_out": L, "Vent": V, "hour_sin": hs, "hour_cos": hc}, index=df.index)
    assert (res.index.to_series().diff().dropna() == pd.Timedelta("10min")).all()
    res.to_parquet(out); print(f"{out}: {len(res)} rows, {res.index.min()} ~ {res.index.max()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); ap.add_argument("--kma", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); main(a.raw, a.kma, a.out)
