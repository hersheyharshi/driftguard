"""The DriftGuard pipeline: map -> coerce -> derive -> inspect -> repair ->
verify -> contracts -> load, with a lineage record for every step."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .contracts import Contracts
from .registry import SchemaRegistry, _compact
from .sentinel import Finding, Sentinel
from .transform import ZoneResolver, coerce_to_canonical, derive, load_schema
from .warehouse import MemoryWarehouse


@dataclass
class Policy:
    repair_units: bool = True
    repair_clock: bool = True
    repair_swaps: bool = True
    drop_duplicate_rows: bool = True


@dataclass
class BatchReport:
    batch_id: str
    batch_date: str
    status: str = "PENDING"   # WARMUP | ACCEPTED | ACCEPTED_REPAIRED | HELD | REJECTED
    rows_in: int = 0
    rows_loaded: int = 0
    rows_quarantined: int = 0
    findings: list = field(default_factory=list)
    repairs: list = field(default_factory=list)
    mapping: dict = field(default_factory=dict)
    lineage: list = field(default_factory=list)

    def checks(self) -> set[str]:
        return {f["check"] for f in self.findings}


def _digest(df: pd.DataFrame) -> str:
    if df is None or not len(df):
        return "empty"
    h = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha1(h.tobytes()).hexdigest()[:12]


class DriftGuard:
    def __init__(self, registry: SchemaRegistry, schema: dict | None = None, contracts: Contracts | None = None,
                 warehouse=None, sentinel: Sentinel | None = None, zones: ZoneResolver | None = None,
                 policy: Policy | None = None, run_id: str | None = None, lineage_digests: bool = True):
        self.schema = schema or load_schema()
        self.registry = registry
        self.contracts = contracts or Contracts()
        self.wh = warehouse if warehouse is not None else MemoryWarehouse()
        self.sentinel = sentinel or Sentinel()
        self.zones = zones
        self.policy = policy or Policy()
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.digests = lineage_digests
        self.reports: list[BatchReport] = []

    def end_warmup(self):
        """Call once after the trusted warm-up batches: calibrates thresholds."""
        self.sentinel.calibrate()

    # ------------------------------------------------------------------ utils
    def row_hashes(self, df: pd.DataFrame) -> np.ndarray:
        keys = [c for c in self.schema["row_key"] if c in df.columns]
        return pd.util.hash_pandas_object(df[keys], index=False).to_numpy()

    def _step(self, rep: BatchReport, name: str, t0: float, rows_in: int, out: pd.DataFrame | None, **notes):
        rep.lineage.append({"run_id": self.run_id, "batch_id": rep.batch_id, "step": name,
                            "ms": round((time.perf_counter() - t0) * 1000, 1), "rows_in": rows_in,
                            "rows_out": None if out is None else len(out),
                            "digest_out": _digest(out) if (self.digests and out is not None) else None,
                            "notes": json.dumps(notes, default=str) if notes else ""})

    # ------------------------------------------------------------------ main
    def process(self, raw: pd.DataFrame, batch_id: str, batch_date, warmup: bool = False) -> BatchReport:
        rep = BatchReport(batch_id=batch_id, batch_date=str(pd.Timestamp(batch_date).date()), rows_in=len(raw))
        n = len(raw)

        t = time.perf_counter()
        m = self.registry.infer(raw)
        rep.mapping = {"mapping": m.mapping, "needs_review": m.needs_review, "ambiguous": m.ambiguous,
                       "resolved_by_invariant": m.resolved_by_invariant, "from_memory": m.from_memory,
                       "missing_required": m.missing_required, "unmapped_sources": m.unmapped_sources}
        self._step(rep, "map_schema", t, n, None, fingerprint=m.fingerprint, review=m.needs_review)

        t = time.perf_counter()
        cdf, coer = coerce_to_canonical(raw, m.mapping, self.schema)
        self._step(rep, "coerce", t, n, cdf, **coer.to_dict())

        t = time.perf_counter()
        df = derive(cdf, self.schema, self.zones)
        hashes = self.row_hashes(df)
        self._step(rep, "derive", t, n, df)

        unknown = {c: sum(coer.unknown_vocab.get(c, {}).values()) / max(n, 1)
                   for c in self.schema["value_maps"] if c in df}

        if warmup:
            clean, q, cstats = self.contracts.apply(df)
            self.sentinel.warmup(clean, batch_date, self.row_hashes(clean), unknown)
            self._load(rep, clean, q, "WARMUP")
            if not m.needs_review and not m.from_memory:
                self.registry.remember(m, approved=True, note="auto: warm-up")
            return self._finish(rep)

        # ---- inspect
        t = time.perf_counter()
        findings = self.sentinel.inspect(df, batch_date, mapping=m, coercion=coer, row_hashes=hashes,
                                         unknown=unknown)
        rep.findings = [f.to_dict() for f in findings]
        self._step(rep, "inspect", t, n, None, checks=[f.check for f in findings])

        if any(f.severity == "reject" for f in findings):
            self._quarantine_batch(rep, df, "batch_rejected:" + ",".join(f.check for f in findings))
            rep.status = "REJECTED"
            return self._finish(rep)

        # ---- repair
        t = time.perf_counter()
        df, repaired, hold = self._repair(df, findings, hashes)
        rep.repairs = repaired
        self._step(rep, "repair", t, n, df, repairs=repaired)
        hold_checks = [f.check for f in findings if f.severity == "hold"] + hold

        # ---- verify repairs by re-inspection
        if repaired and not hold_checks:
            t = time.perf_counter()
            again = self.sentinel.inspect(df, batch_date, check_dups=False)
            still = [f.check for f in again if f.check in ("unit_change", "clock_shift", "column_swap")
                     or f.severity == "hold"]
            h2 = self.row_hashes(df)
            replay = self.sentinel.seen_mask(h2).mean() if len(h2) else 0
            if replay > 0.9:
                still.append("duplicate_batch_after_repair")
            self._step(rep, "verify", t, len(df), None, residual=still)
            if still:
                hold_checks += ["repair_not_verified:" + "+".join(still)]

        if hold_checks:
            self._quarantine_batch(rep, df, "batch_held:" + ",".join(hold_checks))
            rep.status = "HELD"
            return self._finish(rep)

        # ---- contracts + load
        t = time.perf_counter()
        clean, q, cstats = self.contracts.apply(df)
        self._step(rep, "contracts", t, len(df), clean, **cstats)
        self._load(rep, clean, q, "ACCEPTED_REPAIRED" if repaired else "ACCEPTED")
        self.sentinel.update(clean, batch_date, self.row_hashes(clean), unknown)
        if not m.needs_review and not m.from_memory:
            self.registry.remember(m, approved=True, note="auto: high-confidence mapping")
        return self._finish(rep)

    # ------------------------------------------------------------------ helpers
    def _repair(self, df: pd.DataFrame, findings: list[Finding], hashes: np.ndarray):
        df = df.copy()
        done, hold = [], []
        recompute = False
        for f in findings:
            if f.action == "drop_rows" and self.policy.drop_duplicate_rows:
                seen = self.sentinel.seen_mask(hashes)
                dup = seen | pd.Series(hashes).duplicated().to_numpy()
                keep = ~dup
                df = df[keep[: len(df)]].reset_index(drop=True)
                hashes = hashes[keep]
                done.append({"check": f.check, "rows_dropped": int(dup.sum())})
            elif f.check == "unit_change":
                if not self.policy.repair_units:
                    hold.append("unit_change")
                    continue
                k = f.params["factor"]
                for c in f.columns:
                    df[c] = (df[c] / k).round(4)
                done.append({"check": f.check, "columns": f.columns, "divided_by": k})
                recompute = True
            elif f.check == "clock_shift":
                if not self.policy.repair_clock:
                    hold.append("clock_shift")
                    continue
                L = f.params["hours"]
                for c in ("pickup_ts", "dropoff_ts"):
                    df[c] = df[c] - pd.Timedelta(hours=L)
                done.append({"check": f.check, "shifted_hours": -L})
                recompute = True
            elif f.check == "column_swap":
                if not self.policy.repair_swaps:
                    hold.append("column_swap")
                    continue
                a, b = f.params["pair"]
                df[a], df[b] = df[b].to_numpy(), df[a].to_numpy()
                done.append({"check": f.check, "swapped": [a, b]})
                recompute = True
        if recompute:
            base = df.drop(columns=["duration_min", "speed_mph", "total_residual"], errors="ignore")
            df = derive(base, self.schema, self.zones)
        return df, done, hold

    def _quarantine_batch(self, rep: BatchReport, df: pd.DataFrame, reason: str):
        q = df.copy()
        q["reasons"] = reason
        q["batch_id"] = rep.batch_id
        self.wh.append("quarantine", q)
        rep.rows_quarantined = len(q)

    def _load(self, rep: BatchReport, clean: pd.DataFrame, q: pd.DataFrame, status: str):
        t = time.perf_counter()
        clean = clean.assign(batch_id=rep.batch_id)
        self.wh.append("trips", clean.drop(columns=["duration_min", "speed_mph", "total_residual"], errors="ignore"))
        if len(q):
            self.wh.append("quarantine", q.assign(batch_id=rep.batch_id))
        rep.rows_loaded, rep.rows_quarantined, rep.status = len(clean), len(q), status
        self._step(rep, "load", t, len(clean) + len(q), clean)

    def _finish(self, rep: BatchReport) -> BatchReport:
        self.reports.append(rep)
        self.wh.append("batches", pd.DataFrame([{
            "run_id": self.run_id, "batch_id": rep.batch_id, "batch_date": rep.batch_date,
            "status": rep.status, "rows_in": rep.rows_in, "rows_loaded": rep.rows_loaded,
            "rows_quarantined": rep.rows_quarantined, "checks": ",".join(sorted(rep.checks())),
            "repairs": json.dumps(rep.repairs), "mapping": json.dumps(rep.mapping, default=str)}]))
        if rep.findings:
            self.wh.append("findings", pd.DataFrame([{**{k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
                                                          for k, v in f.items()}, "batch_id": rep.batch_id}
                                                       for f in rep.findings]))
        self.wh.append("lineage", pd.DataFrame(rep.lineage))
        return rep


# ---------------------------------------------------------------------------
class NaivePipeline:
    """Baseline: what a typical first pipeline does. Exact-name column matching,
    plain type coercion (bad values -> NaN), no drift checks, no dedup.
    Batches missing a required column fail and are silently skipped."""

    def __init__(self, schema: dict | None = None, zones: ZoneResolver | None = None, warehouse=None):
        self.schema = schema or load_schema()
        self.zones = zones
        self.wh = warehouse if warehouse is not None else MemoryWarehouse()
        self.reports: list[BatchReport] = []
        self.lookup = {}
        for c, spec in self.schema["columns"].items():
            for a in [c] + spec.get("aliases", []):
                self.lookup[_compact(a)] = c

    def process(self, raw: pd.DataFrame, batch_id: str, batch_date, warmup: bool = False) -> BatchReport:
        rep = BatchReport(batch_id, str(pd.Timestamp(batch_date).date()), rows_in=len(raw))
        mapping = {}
        for s in raw.columns:
            c = self.lookup.get(_compact(s))
            if c and c not in mapping:
                mapping[c] = s
        missing = [c for c, sp in self.schema["columns"].items() if sp.get("required") and c not in mapping]
        if missing:
            rep.status = "CRASHED"
            self.reports.append(rep)
            return rep
        out = {}
        for c, s in mapping.items():
            col = raw[s]
            if c in ("pickup_ts", "dropoff_ts"):
                out[c] = pd.to_datetime(col, errors="coerce")
            elif c in self.schema["value_maps"]:
                vm = self.schema["value_maps"][c]
                out[c] = col.map(lambda x: vm.get(str(x).lower(), "unknown"))
            elif c == "store_and_fwd_flag":
                out[c] = col
            else:
                out[c] = pd.to_numeric(col, errors="coerce")
        df = derive(pd.DataFrame(out), self.schema, self.zones)
        df = df[df["pickup_ts"].notna()]
        self.wh.append("trips", df.assign(batch_id=batch_id).drop(columns=["duration_min", "speed_mph", "total_residual"]))
        rep.status, rep.rows_loaded = "ACCEPTED", len(df)
        self.reports.append(rep)
        return rep
