"""Canonicalisation: coerce mapped raw columns into canonical types, derive
helper columns, and compute cross-field invariants."""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources

import numpy as np
import pandas as pd
import yaml

from .profiles import clean_numeric, norm_value, parse_datetime

DATETIME_COLS = {"pickup_ts", "dropoff_ts"}
TEXT_COLS = {"vendor_id", "payment_type", "store_and_fwd_flag"}


def load_yaml(name: str) -> dict:
    return yaml.safe_load(resources.files("driftguard.config").joinpath(name).read_text())


def load_schema() -> dict:
    s = load_yaml("canonical_schema.yaml")
    s["value_maps"] = {c: {str(k).lower(): v for k, v in m.items()}
                       for c, m in s.get("value_maps", {}).items()}
    return s


@dataclass
class CoercionReport:
    repaired: dict = field(default_factory=dict)        # col -> n values repaired
    parse_failures: dict = field(default_factory=dict)  # col -> n values unparseable
    unknown_vocab: dict = field(default_factory=dict)   # col -> {value: count}

    def to_dict(self):
        return {"repaired": self.repaired, "parse_failures": self.parse_failures,
                "unknown_vocab": self.unknown_vocab}


def coerce_to_canonical(raw: pd.DataFrame, mapping: dict[str, str], schema: dict
                        ) -> tuple[pd.DataFrame, CoercionReport]:
    """mapping: canonical column -> raw column."""
    rep = CoercionReport()
    out = {}
    n = len(raw)
    for canon in schema["columns"]:
        src = mapping.get(canon)
        if src is None or src not in raw:
            continue
        s = raw[src]
        if canon in DATETIME_COLS:
            v = parse_datetime(s)
            bad = int((v.isna() & s.notna()).sum())
        elif canon in schema["value_maps"]:
            vm = schema["value_maps"][canon]
            # map on unique values only: fast on multi-million-row files
            lut = {u: norm_value(u) for u in s.dropna().unique()}
            normed = s.map(lut)
            v = normed.map(lambda x: vm.get(x, "unknown") if isinstance(x, str) else None)
            unk = normed[(v == "unknown") & normed.notna() & (normed != "5")]
            if len(unk):
                rep.unknown_vocab[canon] = {str(k): int(c) for k, c in unk.value_counts().head(10).items()}
            bad = 0
        elif canon in TEXT_COLS:
            lut = {u: str(u).strip().upper() for u in s.dropna().unique()}
            v = s.map(lut)
            bad = 0
        else:
            v, n_rep = clean_numeric(s)
            if n_rep:
                rep.repaired[canon] = n_rep
            bad = int((v.isna() & s.notna()).sum())
        if bad:
            rep.parse_failures[canon] = bad
        out[canon] = v.to_numpy() if hasattr(v, "to_numpy") else v
    cdf = pd.DataFrame(out, index=range(n))
    for c in DATETIME_COLS & set(cdf.columns):
        cdf[c] = pd.to_datetime(cdf[c])
    return cdf, rep


class ZoneResolver:
    """Assigns taxi-zone IDs to coordinates (needed for pre-mid-2016 files).

    - synthetic data: nearest-centroid lookup
    - real TLC data: pass the official taxi_zones shapefile (needs geopandas)
    """

    def __init__(self, centroids: pd.DataFrame | None = None, shapefile: str | None = None):
        self.centroids = centroids
        self.shapes = None
        if shapefile:
            import geopandas as gpd  # optional dependency
            self.shapes = gpd.read_file(shapefile).to_crs("EPSG:4326")[["LocationID", "geometry"]]
        if centroids is not None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(centroids[["lon", "lat"]].to_numpy())
            self._ids = centroids["zone"].to_numpy()

    def resolve(self, lon: pd.Series, lat: pd.Series) -> pd.Series:
        ok = lon.notna() & lat.notna()
        out = pd.Series(np.nan, index=lon.index)
        if not ok.any():
            return out
        if self.shapes is not None:
            import geopandas as gpd
            pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon[ok], lat[ok]), crs="EPSG:4326",
                                   index=lon[ok].index)
            j = gpd.sjoin(pts, self.shapes, how="left", predicate="within")
            out[ok] = j.groupby(level=0)["LocationID"].first().reindex(lon[ok].index).to_numpy()
        elif self.centroids is not None:
            _, idx = self._tree.query(np.c_[lon[ok], lat[ok]])
            out[ok] = self._ids[idx]
        return out


def derive(cdf: pd.DataFrame, schema: dict, zones: ZoneResolver | None = None) -> pd.DataFrame:
    df = cdf.copy()
    for z, lo, la in (("pu_zone", "pickup_lon", "pickup_lat"), ("do_zone", "dropoff_lon", "dropoff_lat")):
        if z not in df and lo in df and la in df and zones is not None:
            df[z] = zones.resolve(df[lo], df[la])
    if {"pickup_ts", "dropoff_ts"} <= set(df.columns):
        df["duration_min"] = (df["dropoff_ts"] - df["pickup_ts"]).dt.total_seconds() / 60
    else:
        df["duration_min"] = np.nan
    if "trip_distance_mi" in df:
        hrs = df["duration_min"] / 60
        df["speed_mph"] = np.where(hrs > 0, df["trip_distance_mi"] / hrs.where(hrs > 0), np.nan)
    else:
        df["speed_mph"] = np.nan
    comps = [c for c in schema["total_components"] if c in df]
    if "total_amount" in df and comps:
        df["total_residual"] = df["total_amount"] - df[comps].fillna(0).sum(axis=1)
    else:
        df["total_residual"] = np.nan
    return df


INVARIANTS = ("dropoff_after_pickup", "speed_plausible", "total_consistent")


def invariant_rates(df: pd.DataFrame) -> dict[str, float]:
    """Share of rows satisfying each cross-field invariant (NaN if not computable)."""
    r = {}
    if {"pickup_ts", "dropoff_ts"} <= set(df.columns):
        r["dropoff_after_pickup"] = float((df["dropoff_ts"] > df["pickup_ts"]).mean())
    sp = df.get("speed_mph")
    if sp is not None and sp.notna().any():
        r["speed_plausible"] = float((sp[sp.notna()] <= 80).mean())
    tr = df.get("total_residual")
    if tr is not None and tr.notna().any():
        r["total_consistent"] = float((tr[tr.notna()].abs() <= 0.05).mean())
    return r
