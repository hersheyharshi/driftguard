"""Synthetic NYC-taxi-like trips with era-specific raw schemas.

Why synthetic data at all?  Real data has no labels: you never know which rows
or batches are actually broken. A generator with *known ground truth* lets us
measure exactly what the pipeline catches and misses. The same pipeline then
runs unchanged on real TLC files (see scripts/download_tlc.py).

The six raw "eras" are modelled on how the TLC yellow-taxi files changed:
2009 (Trip_Pickup_DateTime, Start_Lon, "CASH"), 2010 (pickup_datetime, "CSH"),
2015 (tpep_*, numeric codes), mid-2016+ (zone IDs replace coordinates),
2019 (+congestion_surcharge), 2023 (+Airport_fee). Check the official data
dictionaries before claiming exact fidelity for a specific month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_ZONES = 60
AIRPORT_ZONES = (5, 9)  # synthetic stand-ins for JFK / LGA

HOURLY_WD = np.array([0.45, 0.30, 0.20, 0.15, 0.15, 0.25, 0.55, 0.90, 1.05, 1.00, 0.95, 0.97,
                      1.00, 1.00, 1.05, 1.08, 1.05, 1.15, 1.30, 1.28, 1.15, 1.05, 0.90, 0.70])
HOURLY_WE = np.array([0.95, 0.85, 0.70, 0.50, 0.35, 0.25, 0.25, 0.35, 0.50, 0.70, 0.85, 0.95,
                      1.00, 1.00, 1.00, 1.00, 1.00, 1.05, 1.10, 1.10, 1.10, 1.15, 1.15, 1.05])
DOW = np.array([0.92, 0.98, 1.02, 1.05, 1.10, 1.08, 0.90])  # Mon..Sun

CANONICAL_ORDER = [
    "vendor_id", "pickup_ts", "dropoff_ts", "passenger_count", "trip_distance_mi",
    "pickup_lon", "pickup_lat", "dropoff_lon", "dropoff_lat", "pu_zone", "do_zone",
    "rate_code", "store_and_fwd_flag", "payment_type", "fare_amount", "extra", "mta_tax",
    "tip_amount", "tolls_amount", "improvement_surcharge", "congestion_surcharge",
    "airport_fee", "total_amount",
]


def zone_centroids(n: int = N_ZONES, seed: int = 7) -> pd.DataFrame:
    """Jittered grid of zone centroids inside a NYC-like bounding box."""
    rng = np.random.default_rng(seed)
    k = int(np.ceil(np.sqrt(n)))
    xs, ys = np.linspace(-74.02, -73.78, k), np.linspace(40.60, 40.88, k)
    pts = np.array([(x, y) for y in ys for x in xs][:n]) + rng.normal(0, 0.004, (n, 2))
    return pd.DataFrame({"zone": np.arange(1, n + 1), "lon": pts[:, 0], "lat": pts[:, 1]})


class TripGenerator:
    """Generates one day of canonical ('truth') trips at a time."""

    def __init__(self, trips_per_day: int = 6000, n_zones: int = N_ZONES, seed: int = 0):
        self.n = trips_per_day
        self.n_zones = n_zones
        self.rng = np.random.default_rng(seed)
        zr = np.random.default_rng(1234)  # zone character is fixed across seeds
        self.centroids = zone_centroids(n_zones)
        rank = np.arange(1, n_zones + 1)
        self.pu_weight = 1.0 / rank ** 0.9
        self.do_weight = 1.0 / rank ** 0.35
        self.zone_amp = zr.uniform(0.0, 0.8, n_zones)       # how "peaky" each zone is
        self.zone_phase = zr.choice([9.0, 13.0, 19.0, 23.0], n_zones)  # business..nightlife

    def _zone_probs(self, hour: int) -> np.ndarray:
        tilt = 1 + self.zone_amp * np.cos(2 * np.pi * (hour - self.zone_phase) / 24)
        p = self.pu_weight * tilt
        return p / p.sum()

    def day(self, date) -> pd.DataFrame:
        rng = self.rng
        date = pd.Timestamp(date).normalize()
        dow = date.dayofweek
        prof = HOURLY_WE if dow >= 5 else HOURLY_WD
        share = prof / prof.sum()
        n_day = rng.poisson(self.n * DOW[dow])
        counts = rng.multinomial(n_day, share)
        pu_parts, hour_parts = [], []
        for h, c in enumerate(counts):
            if c:
                pu_parts.append(rng.choice(self.n_zones, size=c, p=self._zone_probs(h)) + 1)
                hour_parts.append(np.full(c, h))
        pu = np.concatenate(pu_parts)
        hour = np.concatenate(hour_parts)
        m = len(pu)
        secs = hour * 3600 + rng.integers(0, 3600, m)
        pickup = date + pd.to_timedelta(secs, unit="s")

        do = rng.choice(self.n_zones, size=m, p=self.do_weight / self.do_weight.sum()) + 1
        airport = np.isin(pu, AIRPORT_ZONES) | np.isin(do, AIRPORT_ZONES)
        dist = np.exp(rng.normal(np.log(1.8), 0.75, m)) + airport * rng.normal(12, 2, m)
        dist = np.clip(dist, 0.1, 60).round(2)
        night = ((hour >= 22) | (hour < 6)).astype(float)
        speed = np.clip(10.5 + 6 * night + 5 * airport + rng.normal(0, 2, m), 3, 40)
        dur_min = dist / speed * 60 + rng.uniform(0.5, 3, m)
        dropoff = pickup + pd.to_timedelta((dur_min * 60).round(), unit="s")

        rate = np.where(np.isin(pu, AIRPORT_ZONES[:1]) & (rng.random(m) < 0.7), 2,
                        np.where(rng.random(m) < 0.01, 5, 1))
        fare = np.where(rate == 2, 70.0, 3.0 + 1.75 * dist + 0.35 * dur_min).round(2)
        weekday_rush = (dow < 5) & (hour >= 16) & (hour < 20)
        extra = np.where(weekday_rush, 2.5, np.where(night == 1, 1.0, 0.0))
        pay = rng.choice(["card", "cash", "no_charge", "dispute"], m, p=[0.72, 0.25, 0.02, 0.01])
        tip = np.where(pay == "card", np.clip(fare * rng.normal(0.2, 0.05, m), 0, None), 0).round(2)
        tolls = np.where(rng.random(m) < 0.06 + 0.4 * airport, 6.94, 0.0)
        cong = np.where(rng.random(m) < 0.75, 2.5, 0.0)
        apfee = np.where(np.isin(pu, AIRPORT_ZONES), 1.75, 0.0)
        pax = rng.choice([0, 1, 2, 3, 4, 5, 6], m, p=[0.01, 0.69, 0.14, 0.05, 0.03, 0.05, 0.03])

        c = self.centroids.set_index("zone")
        df = pd.DataFrame({
            "vendor_id": rng.choice(["cmt", "vts"], m, p=[0.35, 0.65]),
            "pickup_ts": pickup,
            "dropoff_ts": dropoff,
            "passenger_count": pax.astype(float),
            "trip_distance_mi": dist,
            "pickup_lon": (c.loc[pu, "lon"].to_numpy() + rng.normal(0, 0.002, m)).round(6),
            "pickup_lat": (c.loc[pu, "lat"].to_numpy() + rng.normal(0, 0.002, m)).round(6),
            "dropoff_lon": (c.loc[do, "lon"].to_numpy() + rng.normal(0, 0.002, m)).round(6),
            "dropoff_lat": (c.loc[do, "lat"].to_numpy() + rng.normal(0, 0.002, m)).round(6),
            "pu_zone": pu.astype(float),
            "do_zone": do.astype(float),
            "rate_code": rate.astype(float),
            "store_and_fwd_flag": np.where(rng.random(m) < 0.005, "Y", "N"),
            "payment_type": pay,
            "fare_amount": fare,
            "extra": extra,
            "mta_tax": 0.5,
            "tip_amount": tip,
            "tolls_amount": tolls,
            "improvement_surcharge": 1.0,
            "congestion_surcharge": cong,
            "airport_fee": apfee,
        })
        df["total_amount"] = df[["fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
                                 "improvement_surcharge", "congestion_surcharge",
                                 "airport_fee"]].sum(axis=1).round(2)
        return df.sort_values("pickup_ts", ignore_index=True)


# --------------------------------------------------------------------------
# Raw "eras": canonical name -> raw name, plus value encodings per era.
# --------------------------------------------------------------------------
_PAY_2009 = {"card": "Credit", "cash": "CASH", "no_charge": "No Charge", "dispute": "Dispute"}
_PAY_2010 = {"card": "CRD", "cash": "CSH", "no_charge": "NOC", "dispute": "DIS"}
_PAY_CODE = {"card": 1, "cash": 2, "no_charge": 3, "dispute": 4}
_VENDOR_CODE = {"cmt": 1, "vts": 2}

ERAS: dict[str, dict] = {
    "2009": dict(names={
        "vendor_id": "vendor_name", "pickup_ts": "Trip_Pickup_DateTime",
        "dropoff_ts": "Trip_Dropoff_DateTime", "passenger_count": "Passenger_Count",
        "trip_distance_mi": "Trip_Distance", "pickup_lon": "Start_Lon", "pickup_lat": "Start_Lat",
        "rate_code": "Rate_Code", "store_and_fwd_flag": "store_and_forward",
        "dropoff_lon": "End_Lon", "dropoff_lat": "End_Lat", "payment_type": "Payment_Type",
        "fare_amount": "Fare_Amt", "extra": "surcharge", "mta_tax": "mta_tax",
        "tip_amount": "Tip_Amt", "tolls_amount": "Tolls_Amt", "total_amount": "Total_Amt"},
        pay=_PAY_2009, vendor="upper", ts_as_text=True, improvement=None),
    "2010": dict(names={
        "vendor_id": "vendor_id", "pickup_ts": "pickup_datetime", "dropoff_ts": "dropoff_datetime",
        "passenger_count": "passenger_count", "trip_distance_mi": "trip_distance",
        "pickup_lon": "pickup_longitude", "pickup_lat": "pickup_latitude", "rate_code": "rate_code",
        "store_and_fwd_flag": "store_and_fwd_flag", "dropoff_lon": "dropoff_longitude",
        "dropoff_lat": "dropoff_latitude", "payment_type": "payment_type",
        "fare_amount": "fare_amount", "extra": "surcharge", "mta_tax": "mta_tax",
        "tip_amount": "tip_amount", "tolls_amount": "tolls_amount", "total_amount": "total_amount"},
        pay=_PAY_2010, vendor="upper", ts_as_text=True, improvement=None),
}
_MODERN = {
    "vendor_id": "VendorID", "pickup_ts": "tpep_pickup_datetime",
    "dropoff_ts": "tpep_dropoff_datetime", "passenger_count": "passenger_count",
    "trip_distance_mi": "trip_distance", "rate_code": "RatecodeID",
    "store_and_fwd_flag": "store_and_fwd_flag", "payment_type": "payment_type",
    "fare_amount": "fare_amount", "extra": "extra", "mta_tax": "mta_tax",
    "tip_amount": "tip_amount", "tolls_amount": "tolls_amount",
    "improvement_surcharge": "improvement_surcharge", "total_amount": "total_amount"}
_COORDS = {"pickup_lon": "pickup_longitude", "pickup_lat": "pickup_latitude",
           "dropoff_lon": "dropoff_longitude", "dropoff_lat": "dropoff_latitude"}
_ZONES = {"pu_zone": "PULocationID", "do_zone": "DOLocationID"}

ERAS["2015"] = dict(names={**_MODERN, **_COORDS}, pay=_PAY_CODE, vendor="code",
                    ts_as_text=False, improvement=0.3)
ERAS["2017"] = dict(names={**_MODERN, **_ZONES}, pay=_PAY_CODE, vendor="code",
                    ts_as_text=False, improvement=0.3)
ERAS["2019"] = dict(names={**_MODERN, **_ZONES, "congestion_surcharge": "congestion_surcharge"},
                    pay=_PAY_CODE, vendor="code", ts_as_text=False, improvement=0.3)
ERAS["2023"] = dict(names={**_MODERN, **_ZONES, "congestion_surcharge": "congestion_surcharge",
                           "airport_fee": "Airport_fee"},
                    pay=_PAY_CODE, vendor="code", ts_as_text=False, improvement=1.0)

MODERN_ERA = "2023"


def era_truth(truth: pd.DataFrame, era: str) -> pd.DataFrame:
    """Canonical truth as it *should* look after correct ingestion of this era
    (columns the era lacks are dropped, totals recomputed)."""
    spec = ERAS[era]
    t = truth.copy()
    if spec["improvement"] is not None:
        t["improvement_surcharge"] = spec["improvement"]
    keep = list(spec["names"].keys())
    comps = [c for c in ["fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
                         "improvement_surcharge", "congestion_surcharge", "airport_fee"] if c in keep]
    t["total_amount"] = t[comps].sum(axis=1).round(2)
    return t[[c for c in CANONICAL_ORDER if c in keep]]


def to_raw(truth: pd.DataFrame, era: str = MODERN_ERA) -> pd.DataFrame:
    """Render canonical truth in the raw layout of a given era."""
    spec = ERAS[era]
    t = era_truth(truth, era)
    if spec["vendor"] == "upper":
        t["vendor_id"] = t["vendor_id"].str.upper()
    else:
        t["vendor_id"] = t["vendor_id"].map(_VENDOR_CODE).astype("int64")
    t["payment_type"] = t["payment_type"].map(spec["pay"])
    if spec["ts_as_text"]:
        for c in ("pickup_ts", "dropoff_ts"):
            t[c] = t[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in ("pu_zone", "do_zone"):
        if c in t:
            t[c] = t[c].astype("int64")
    return t.rename(columns=spec["names"])[list(spec["names"].values())]
