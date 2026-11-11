"""Batch-level sentinel: detects that a batch is wrong AND says *why*.

Most drift monitors stop at "column X drifted (KS=0.4)". The sentinel goes one
step further and tests specific, repairable hypotheses:

  unit_change      batch / k matches the reference for a known factor k
                   (1.609 mi->km, 100 dollars->cents, ...)
  clock_shift      hour-of-day profile matches the reference after a circular
                   shift of L hours (timezone / DST bugs)
  column_swap      column A now looks like reference B and vice versa
  duplicate_batch  the batch was already delivered (replay)
  partial_dups     a share of rows was already delivered (overlapping windows)
  null_burst, volume_anomaly, missing_column, vocabulary_drift,
  invariant_break, late_arrivals, type_repair
  distribution_drift   drift that matches no known cause: held for a human

Thresholds are *self-calibrating*: during a warm-up window of trusted batches
the sentinel measures its own statistics' natural day-to-day noise and sets
alarm thresholds at mean + k*std (with sane floors), per column.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .transform import invariant_rates

UNIT_FACTORS = {1.609344: "miles->km", 0.621371: "km->miles", 100.0: "units->cents",
                0.01: "cents->units", 1000.0: "x1000 (m/km, ms/s)", 0.001: "/1000",
                60.0: "x60 (min->s, h->min)", 1 / 60: "/60"}
MONITORED = ["trip_distance_mi", "fare_amount", "tip_amount", "tolls_amount", "total_amount",
             "extra", "mta_tax", "improvement_surcharge", "congestion_surcharge", "airport_fee",
             "passenger_count", "duration_min", "pu_zone"]
NULL_WATCH = ["pickup_ts", "dropoff_ts", "passenger_count", "trip_distance_mi", "pu_zone",
              "do_zone", "fare_amount", "total_amount", "payment_type"]


@dataclass
class Finding:
    check: str
    severity: str                  # info | warn | hold | reject
    columns: list = field(default_factory=list)
    statistic: float | None = None
    threshold: float | None = None
    cause: str = ""
    params: dict = field(default_factory=dict)
    action: str = "none"           # none | repair | drop_rows | hold | reject

    def to_dict(self):
        return asdict(self)


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 20 or len(b) < 20:
        return 0.0
    return float(ks_2samp(a, b).statistic)


def _loc(a: np.ndarray, b: np.ndarray) -> float:
    """Shift in mean log-value: far more sensitive than KS to small scale changes."""
    if len(a) < 20 or len(b) < 20:
        return 0.0
    return float(abs(np.mean(np.log1p(np.abs(a))) - np.mean(np.log1p(np.abs(b)))))


def _daytype(d) -> str:
    return "weekend" if pd.Timestamp(d).dayofweek >= 5 else "weekday"


def _hour_hist(ts: pd.Series) -> np.ndarray:
    h = np.bincount(ts.dropna().dt.hour.to_numpy(), minlength=24).astype(float)
    return h / max(h.sum(), 1)


class Sentinel:
    def __init__(self, window: int = 14, sample: int = 3000, k_sigma: float = 5.0, seed: int = 0):
        self.window, self.sample, self.k = window, sample, k_sigma
        self.rng = np.random.default_rng(seed)
        self.ref_vals: dict[str, deque] = {}
        self.ref_hours: dict[str, deque] = {"weekday": deque(maxlen=window), "weekend": deque(maxlen=window)}
        self.ref_counts: dict[str, deque] = {"weekday": deque(maxlen=window), "weekend": deque(maxlen=window)}
        self.ref_nulls: dict[str, deque] = {}
        self.ref_inv: dict[str, deque] = {}
        self.ref_unk: dict[str, deque] = {}
        self.present: set[str] = set()
        # row hashes of the last `window` accepted batches (bounded memory: real
        # months are ~3M rows; replays almost always concern recent deliveries)
        self.hash_window: deque = deque(maxlen=window)
        self.th: dict[str, float] = {}
        self._warm: list[tuple[pd.DataFrame, pd.Timestamp]] = []
        self.calibrated = False

    # ------------------------------------------------------------ reference
    def seen_mask(self, hashes: np.ndarray) -> np.ndarray:
        if not self.hash_window or not len(hashes):
            return np.zeros(len(hashes), bool)
        return np.isin(hashes, np.concatenate(list(self.hash_window)))

    def _continuous(self, c: str) -> bool:
        v = self.ref_vals.get(c)
        return bool(v) and len(np.unique(v[0])) > 30 and np.nanmin(v[0]) >= 0

    def _sample(self, x: np.ndarray) -> np.ndarray:
        x = x[~np.isnan(x)]
        return self.rng.choice(x, self.sample, replace=False) if len(x) > self.sample else x

    def update(self, df: pd.DataFrame, batch_date, row_hashes: np.ndarray | None = None,
               unknown: dict | None = None):
        """Add an accepted batch to the rolling reference.
        unknown: column -> share of values outside the known vocabulary."""
        dt = _daytype(batch_date)
        for c, v in (unknown or {}).items():
            self.ref_unk.setdefault(c, deque(maxlen=self.window)).append(v)
        for c in MONITORED:
            if c in df and df[c].notna().mean() > 0.5:
                self.ref_vals.setdefault(c, deque(maxlen=self.window)).append(
                    self._sample(df[c].to_numpy(dtype=float)))
                self.present.add(c)
        for c in NULL_WATCH:
            if c in df:
                self.ref_nulls.setdefault(c, deque(maxlen=self.window)).append(float(df[c].isna().mean()))
        if "pickup_ts" in df:
            self.ref_hours[dt].append(_hour_hist(df["pickup_ts"]))
        self.ref_counts[dt].append(len(df))
        for k, v in invariant_rates(df).items():
            self.ref_inv.setdefault(k, deque(maxlen=self.window)).append(v)
        if row_hashes is not None:
            self.hash_window.append(np.unique(row_hashes))

    def warmup(self, df: pd.DataFrame, batch_date, row_hashes=None, unknown: dict | None = None):
        self._warm.append((df, pd.Timestamp(batch_date)))
        self.update(df, batch_date, row_hashes, unknown)

    def calibrate(self):
        """Leave-one-out noise measurement over the warm-up window -> thresholds."""
        stats: dict[str, list] = {}
        n = len(self._warm)
        for i, (df, d) in enumerate(self._warm):
            for c in sorted(self.present):
                ref = np.concatenate([v for j, v in enumerate(self.ref_vals[c]) if j != i])
                xs = self._sample(df[c].to_numpy(dtype=float))
                stats.setdefault(f"ks:{c}", []).append(_ks(xs, ref))
                if self._continuous(c):
                    stats.setdefault(f"loc:{c}", []).append(_loc(xs, ref))
            dt = _daytype(d)
            others = [h for (df2, d2) in [w for k, w in enumerate(self._warm) if k != i] if _daytype(d2) == dt
                      for h in [_hour_hist(df2["pickup_ts"])]]
            if others:
                stats.setdefault("hour", []).append(float(np.abs(_hour_hist(df["pickup_ts"]) - np.mean(others, 0)).sum()))
            cnts = [len(df2) for k, (df2, d2) in enumerate(self._warm) if k != i and _daytype(d2) == dt]
            if cnts:
                stats.setdefault("volume", []).append(abs(len(df) / np.median(cnts) - 1))
            for c in NULL_WATCH:
                if c in df:
                    stats.setdefault(f"null:{c}", []).append(float(df[c].isna().mean()))
            for k, v in invariant_rates(df).items():
                stats.setdefault(f"inv:{k}", []).append(v)
        for c, vals in self.ref_unk.items():
            stats[f"unk:{c}"] = list(vals)
        floors = {"ks": 0.05, "loc": 0.03, "hour": 0.10, "volume": 0.20, "null": 0.02, "inv": 0.02, "unk": 0.005}
        for key, vals in stats.items():
            v = np.asarray(vals)
            kind = key.split(":")[0]
            spread = self.k * v.std(ddof=1) if len(v) > 1 else 0
            if kind in ("ks", "loc", "hour", "volume"):
                self.th[key] = max(floors[kind], float(v.mean() + spread))
            else:  # null / inv: allowed deviation from the mean
                self.th[key] = max(floors[kind], float(spread))
        self.calibrated = True
        self._warm = []

    # ------------------------------------------------------------ inspection
    def inspect(self, df: pd.DataFrame, batch_date, mapping=None, coercion=None,
                row_hashes: np.ndarray | None = None, check_dups: bool = True,
                unknown: dict | None = None) -> list[Finding]:
        F: list[Finding] = []
        d = pd.Timestamp(batch_date)
        dt = _daytype(d)

        # ---- duplicates / replays
        n_dup_seen = n_dup_within = 0
        if check_dups and row_hashes is not None and len(row_hashes):
            seen = self.seen_mask(row_hashes)
            frac = float(seen.mean())
            n_dup_within = int(pd.Series(row_hashes).duplicated().sum())
            if frac > 0.9:
                return [Finding("duplicate_batch", "reject", statistic=frac, threshold=0.9,
                                cause="batch was already delivered (replay)", action="reject")]
            n_dup_seen = int(seen.sum())
            if n_dup_seen + n_dup_within > max(5, 0.005 * len(df)):
                F.append(Finding("partial_duplicates", "warn", statistic=(n_dup_seen + n_dup_within) / len(df),
                                 threshold=0.005, cause="rows already delivered / repeated in batch",
                                 params={"seen_before": n_dup_seen, "within_batch": n_dup_within},
                                 action="drop_rows"))

        # ---- schema-level
        if mapping is not None:
            missing = [c for c in sorted(self.present) if c not in df.columns or df[c].isna().all()]
            if missing:
                F.append(Finding("missing_column", "warn", columns=sorted(missing),
                                 cause="column present in reference is absent"))
            if mapping.ambiguous:
                F.append(Finding("ambiguous_mapping", "hold", columns=[c for p in mapping.ambiguous for c in p],
                                 cause="column identity could not be established", action="hold",
                                 params={"pairs": mapping.ambiguous}))
            if mapping.missing_required:
                F.append(Finding("missing_required", "hold", columns=mapping.missing_required,
                                 cause="required canonical column not found", action="hold"))
        if coercion is not None:
            if coercion.repaired:
                F.append(Finding("type_repair", "info", columns=list(coercion.repaired),
                                 params=coercion.repaired, cause="formatted numbers repaired ('$1,234.50')"))
            for c, share in (unknown or {}).items():
                base = float(np.mean(self.ref_unk[c])) if self.ref_unk.get(c) else 0.0
                th = self.th.get(f"unk:{c}", 0.005)
                if share - base > th:
                    F.append(Finding("vocabulary_drift", "warn", columns=[c], statistic=share, threshold=base + th,
                                     params=coercion.unknown_vocab.get(c, {}),
                                     cause=f"unseen values in {c}: {base:.1%} -> {share:.1%} of rows"))

        # ---- volume (after removing duplicates)
        cnts = list(self.ref_counts[dt])
        if cnts:
            dev = (len(df) - n_dup_seen - n_dup_within) / np.median(cnts) - 1
            th = self.th.get("volume", 0.25)
            if abs(dev) > th:
                F.append(Finding("volume_anomaly", "warn", statistic=float(dev), threshold=th,
                                 cause="row count far from reference"))

        # ---- null bursts
        for c in NULL_WATCH:
            if c in df and c in self.ref_nulls and self.ref_nulls[c]:
                base = float(np.mean(self.ref_nulls[c]))
                rate = float(df[c].isna().mean())
                th = self.th.get(f"null:{c}", 0.02)
                if rate - base > th and rate < 0.999:
                    F.append(Finding("null_burst", "warn", columns=[c], statistic=rate, threshold=base + th,
                                     cause=f"null rate jumped {base:.1%} -> {rate:.1%}"))

        # ---- temporal: clock shift
        if "pickup_ts" in df and self.ref_hours[dt]:
            ref_h = np.mean(self.ref_hours[dt], 0)
            h = _hour_hist(df["pickup_ts"])
            d0 = float(np.abs(h - ref_h).sum())
            th = self.th.get("hour", 0.15)
            if d0 > th:
                lags = {L: float(np.abs(np.roll(h, -L) - ref_h).sum()) for L in range(-12, 13) if L}
                L, dl = min(lags.items(), key=lambda kv: kv[1])
                if dl < 0.5 * d0 and dl <= 1.5 * th:
                    F.append(Finding("clock_shift", "warn", columns=["pickup_ts", "dropoff_ts"],
                                     statistic=d0, threshold=th, cause=f"timestamps shifted by {L:+d}h",
                                     params={"hours": int(L)}, action="repair"))
                else:
                    F.append(Finding("temporal_drift", "warn", columns=["pickup_ts"], statistic=d0,
                                     threshold=th, cause="hour-of-day profile changed"))
            # late arrivals (rows whose pickup date isn't this batch's date)
            dates = df["pickup_ts"].dt.normalize()
            late = float((dates.notna() & (dates != d.normalize())).mean())
            if late > 0.02 and not any(f.check == "clock_shift" for f in F):
                F.append(Finding("late_arrivals", "info", statistic=late, threshold=0.02,
                                 cause="rows belong to other dates; routed by event time"))

        # ---- per-column distribution drift + attribution
        drifted = {}
        for c in sorted(self.present):
            if c not in df or df[c].notna().mean() < 0.5:
                continue
            x = self._sample(df[c].to_numpy(dtype=float))
            ref = np.concatenate(self.ref_vals[c])
            ks = _ks(x, ref)
            th = self.th.get(f"ks:{c}", 0.06)
            loc = _loc(x, ref) if self._continuous(c) else 0.0
            if ks > th or loc > self.th.get(f"loc:{c}", 0.05):
                drifted[c] = (x, ks, th)
        attributed = set()
        units: dict[float, list] = {}
        for c, (x, ks, th) in drifted.items():
            ref = np.concatenate(self.ref_vals[c])
            best_k, best = None, ks
            for k in UNIT_FACTORS:
                s = _ks(x / k, ref)
                if s < best:
                    best_k, best = k, s
            if best_k is not None and best <= max(1.5 * th, 0.08) and best < 0.5 * ks:
                units.setdefault(best_k, []).append((c, ks))
                attributed.add(c)
        for k, cols in units.items():
            F.append(Finding("unit_change", "warn", columns=[c for c, _ in cols],
                             statistic=max(s for _, s in cols), cause=f"values scaled by {k:g} ({UNIT_FACTORS[k]})",
                             params={"factor": k}, action="repair"))
        rest = [c for c in drifted if c not in attributed]
        for i, a in enumerate(rest):
            for b in rest[i + 1:]:
                if a in attributed or b in attributed:
                    continue
                xa, _, tha = drifted[a]
                xb, _, thb = drifted[b]
                ra, rb = np.concatenate(self.ref_vals[a]), np.concatenate(self.ref_vals[b])
                if _ks(xa, rb) <= max(1.5 * thb, 0.08) and _ks(xb, ra) <= max(1.5 * tha, 0.08):
                    F.append(Finding("column_swap", "warn", columns=[a, b], cause=f"{a} and {b} values swapped",
                                     params={"pair": [a, b]}, action="repair"))
                    attributed |= {a, b}
        for c in rest:
            if c in attributed:
                continue
            x, ks, th = drifted[c]
            ref = np.concatenate(self.ref_vals[c])
            scale = float(np.median(x) / np.median(ref)) if np.median(ref) else None
            sev = "hold" if ks > 3 * th else "warn"
            F.append(Finding("distribution_drift", sev, columns=[c], statistic=ks, threshold=th,
                             cause="no known cause matched" + (f"; median ratio {scale:.3f}" if scale else ""),
                             params={"median_ratio": scale}, action="hold" if sev == "hold" else "none"))

        # ---- cross-field invariants
        for k, v in invariant_rates(df).items():
            if k in self.ref_inv and self.ref_inv[k]:
                base = float(np.mean(self.ref_inv[k]))
                th = self.th.get(f"inv:{k}", 0.02)
                if base - v > th:
                    F.append(Finding("invariant_break", "warn", columns=[k], statistic=v, threshold=base - th,
                                     cause=f"{k} pass rate fell {base:.1%} -> {v:.1%}"))
        return F
