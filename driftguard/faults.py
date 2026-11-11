"""Fault injector: realistic, *labelled* data failures applied to raw batches.

Each fault returns the corrupted raw frame and the checks a good pipeline is
expected to raise. Faults operate on the modern (2023-era) raw layout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# fault -> (expected checks: any of these counts as "detected", correct cause)
EXPECTED = {
    "header_synonyms":   {"detect": set(), "cause": None},               # benign: must map silently
    "obfuscated_headers": {"detect": set(), "cause": None},              # judged by mapping accuracy
    "unit_km":           {"detect": {"unit_change", "distribution_drift", "invariant_break"}, "cause": "unit_change"},
    "cents":             {"detect": {"unit_change", "distribution_drift", "invariant_break"}, "cause": "unit_change"},
    "duplicate_batch":   {"detect": {"duplicate_batch"}, "cause": "duplicate_batch"},
    "partial_duplicates": {"detect": {"partial_duplicates"}, "cause": "partial_duplicates"},
    "null_burst":        {"detect": {"null_burst"}, "cause": "null_burst"},
    "clock_shift":       {"detect": {"clock_shift", "temporal_drift"}, "cause": "clock_shift"},
    "type_corruption":   {"detect": {"type_repair"}, "cause": "type_repair"},
    "column_swap":       {"detect": {"column_swap", "distribution_drift"}, "cause": "column_swap"},
    "subtle_scale":      {"detect": {"distribution_drift", "unit_change"}, "cause": "distribution_drift"},
    "dropped_column":    {"detect": {"missing_column"}, "cause": "missing_column"},
}
FAULTS = list(EXPECTED)
DETECTABLE = [f for f in FAULTS if EXPECTED[f]["detect"]]

SYNONYMS = {"tpep_pickup_datetime": "pickup_datetime", "tpep_dropoff_datetime": "dropoff_datetime",
            "VendorID": "vendor_id", "RatecodeID": "rate_code", "PULocationID": "pickup_zone",
            "DOLocationID": "dropoff_location", "Airport_fee": "airport_charge",
            "fare_amount": "fare", "tip_amount": "tip", "total_amount": "total_amt",
            "trip_distance": "trip_dist"}


def _c(df: pd.DataFrame, name: str) -> str:
    """Actual column name, matched case-insensitively (real files vary: Airport_fee/airport_fee)."""
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    raise KeyError(name)


def inject(fault: str, raw: pd.DataFrame, rng: np.random.Generator,
           previous: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    p: dict = {}
    if fault == "header_synonyms":
        low = {k.lower(): v for k, v in SYNONYMS.items()}
        df = df.rename(columns={c: low[c.lower()] for c in df.columns if c.lower() in low})
    elif fault == "obfuscated_headers":
        cols = list(df.columns)
        perm = rng.permutation(len(cols))
        df = df.iloc[:, perm]
        p["truth"] = {cols[j]: f"col_{k:02d}" for k, j in enumerate(perm)}
        df.columns = [f"col_{k:02d}" for k in range(len(cols))]
    elif fault == "unit_km":
        c = _c(df, "trip_distance")
        df[c] = (df[c].astype("float64") * 1.609344).round(2)
    elif fault == "cents":
        for n in ("fare_amount", "tip_amount", "total_amount"):
            c = _c(df, n)
            df[c] = (df[c].astype("float64") * 100).round(0)
    elif fault == "duplicate_batch":
        assert previous is not None
        df = previous.copy()   # the feed re-sent yesterday's file instead of today's
    elif fault == "partial_duplicates":
        assert previous is not None
        extra = previous.sample(frac=0.2, random_state=int(rng.integers(1 << 31)))
        df = pd.concat([df, extra], ignore_index=True)
        p["rows_added"] = len(extra)
    elif fault == "null_burst":
        col = _c(df, str(rng.choice(["passenger_count", "trip_distance", "fare_amount", "PULocationID"])))
        mask = rng.random(len(df)) < 0.35
        df[col] = df[col].astype("float64")
        df.loc[mask, col] = np.nan
        p["column"] = str(col)
    elif fault == "clock_shift":
        for n in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
            c = _c(df, n)
            df[c] = df[c] + pd.Timedelta(hours=5)
        p["hours"] = 5
    elif fault == "type_corruption":
        for n in ("fare_amount", "total_amount"):
            c = _c(df, n)
            df[c] = df[c].map(lambda v: f"${v:,.2f}" if pd.notna(v) else None)
    elif fault == "column_swap":
        a, b = _c(df, "tip_amount"), _c(df, "tolls_amount")
        df[a], df[b] = df[b].to_numpy(), df[a].to_numpy()
    elif fault == "subtle_scale":
        # an 8% fare change, internally consistent (totals updated): looks like a real tariff change
        f, t = _c(df, "fare_amount"), _c(df, "total_amount")
        delta = (df[f].astype("float64") * 0.08).round(2)
        df[f] = df[f] + delta
        df[t] = df[t] + delta
    elif fault == "dropped_column":
        df = df.drop(columns=[_c(df, "congestion_surcharge")])
    else:
        raise ValueError(fault)
    return df, {"fault": fault, **p}
