"""Downstream task: day-ahead hourly pickup demand per zone.

Used to answer the question that matters: *how much does bad data hurt a
model, and how much of that damage does the pipeline prevent?*
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def hourly_panel(trips: pd.DataFrame, days: pd.DatetimeIndex, n_zones: int,
                 unknown_days: set | None = None) -> dict[str, np.ndarray]:
    """Wide (hours x zones) arrays of counts, mean distance, mean fare.
    Days in `unknown_days` (known-missing partitions) become NaN, not zero."""
    start = days[0]
    H = len(days) * 24
    cnt = np.zeros((H, n_zones))
    dist = np.full((H, n_zones), np.nan)
    fare = np.full((H, n_zones), np.nan)
    if len(trips):
        t = trips[trips["pickup_ts"].notna() & trips["pu_zone"].notna()]
        h = ((t["pickup_ts"] - start).dt.total_seconds() // 3600).astype("int64").to_numpy()
        z = t["pu_zone"].to_numpy().astype(int) - 1
        ok = (h >= 0) & (h < H) & (z >= 0) & (z < n_zones)
        w = t["n"].to_numpy(dtype=float) if "n" in t else np.ones(len(t))   # pre-aggregated input
        dv = t["trip_distance_mi"].to_numpy(dtype=float)
        fv = t["fare_amount"].to_numpy(dtype=float)
        # means ignore missing values (weights only where the value exists)
        g = pd.DataFrame({"h": h[ok], "z": z[ok], "w": w[ok],
                          "d": np.nan_to_num(dv * w)[ok], "wd": (w * ~np.isnan(dv))[ok],
                          "f": np.nan_to_num(fv * w)[ok], "wf": (w * ~np.isnan(fv))[ok]}).groupby(["h", "z"]).sum()
        a = g.reset_index()
        cnt[a["h"], a["z"]] = a["w"]
        with np.errstate(invalid="ignore", divide="ignore"):
            dist[a["h"], a["z"]] = np.where(a["wd"] > 0, a["d"] / a["wd"], np.nan)
            fare[a["h"], a["z"]] = np.where(a["wf"] > 0, a["f"] / a["wf"], np.nan)
    for d in unknown_days or ():
        i = (pd.Timestamp(d) - start).days
        if 0 <= i < len(days):
            cnt[i * 24:(i + 1) * 24] = np.nan
            dist[i * 24:(i + 1) * 24] = np.nan
            fare[i * 24:(i + 1) * 24] = np.nan
    return {"count": cnt, "dist": dist, "fare": fare}


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def features(panel: dict, days: pd.DatetimeIndex) -> pd.DataFrame:
    c = panel["count"]
    H, Z = c.shape
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        roll = np.nanmean(np.stack([_shift(c, k) for k in range(24, 48)]), axis=0)
    ts = days[0] + pd.to_timedelta(np.arange(H), unit="h")
    f = {
        "lag24": _shift(c, 24), "lag48": _shift(c, 48), "lag168": _shift(c, 168), "roll_d1": roll,
        "dist_lag24": _shift(panel["dist"], 24), "fare_lag24": _shift(panel["fare"], 24),
        "hour": np.repeat(ts.hour.to_numpy()[:, None], Z, 1).astype(float),
        "dow": np.repeat(ts.dayofweek.to_numpy()[:, None], Z, 1).astype(float),
        "zone": np.repeat(np.arange(Z)[None, :], H, 0).astype(float),
    }
    df = pd.DataFrame({k: v.ravel() for k, v in f.items()})
    df["hour_idx"] = np.repeat(np.arange(H), Z)
    return df


def evaluate(cond_trips: pd.DataFrame, truth_trips: pd.DataFrame, days: pd.DatetimeIndex, n_zones: int,
             train_days: tuple[int, int], test_days: tuple[int, int], unknown_days: set | None = None,
             seed: int = 0) -> dict:
    """Train on the condition's own warehouse; score test hours against ground truth."""
    p_cond = hourly_panel(cond_trips, days, n_zones, unknown_days)
    p_true = hourly_panel(truth_trips, days, n_zones)
    X = features(p_cond, days)
    y_cond = p_cond["count"].ravel()
    y_true = p_true["count"].ravel()
    hi = X.pop("hour_idx").to_numpy()
    tr = (hi >= train_days[0] * 24) & (hi < train_days[1] * 24) & ~np.isnan(y_cond)
    te = (hi >= test_days[0] * 24) & (hi < test_days[1] * 24)
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=250, learning_rate=0.06,
                                          # HGB supports <=255 categories; NYC has 265 zones
                                          categorical_features=[X.columns.get_loc("zone")] if n_zones <= 250 else None,
                                          random_state=seed)
    model.fit(X[tr], y_cond[tr])
    pred = model.predict(X[te])
    err = np.abs(pred - y_true[te])
    return {"wape": float(err.sum() / y_true[te].sum()), "mae": float(err.mean()),
            "train_rows": int(tr.sum()), "test_rows": int(te.sum())}
