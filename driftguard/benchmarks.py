"""The three experiments behind the write-up.

E1  Schema mapping: how accurately are columns identified across 6 header
    eras, with real names vs. meaningless names? Ablation over evidence types.
E2  Fault detection & attribution: balanced schedule of 12 labelled fault
    types; recall, correct-cause rate, false alarms on clean batches, and how
    close repaired batches are to ground truth.
E3  Downstream impact: forecast error (WAPE) of a demand model trained on the
    warehouse built by a naive pipeline vs. DriftGuard vs. an oracle, as the
    fault rate rises.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .faults import EXPECTED, FAULTS, inject
from .forecast import evaluate
from .pipeline import DriftGuard, NaivePipeline
from .registry import SchemaRegistry
from .synth import ERAS, MODERN_ERA, N_ZONES, TripGenerator, to_raw, zone_centroids
from .transform import ZoneResolver, load_schema

ABLATIONS = {
    "name only": dict(use_name=True, use_stats=False, use_invariants=False),
    "stats only": dict(use_name=False, use_stats=True, use_invariants=False),
    "name + stats": dict(use_name=True, use_stats=True, use_invariants=False),
    "name + stats + invariants": dict(use_name=True, use_stats=True, use_invariants=True),
}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- E1
def schema_benchmark(seeds=(0, 1, 2, 3, 4), rows=4000) -> pd.DataFrame:
    schema = load_schema()
    out = []
    for seed in seeds:
        g = TripGenerator(rows, seed=100 + seed)
        reg = SchemaRegistry(schema).fit_reference(g.day("2024-02-05"))
        rng = np.random.default_rng(seed)
        for era, spec in ERAS.items():
            raw = to_raw(g.day("2024-02-06"), era)
            truth = dict(spec["names"])
            variants = {"named headers": (raw, truth)}
            cols = list(raw.columns)
            perm = rng.permutation(len(cols))
            ob = raw.iloc[:, perm].copy()
            ren = {cols[j]: f"col_{k:02d}" for k, j in enumerate(perm)}
            ob.columns = [ren[c] for c in ob.columns]
            variants["meaningless headers"] = (ob, {c: ren[r] for c, r in truth.items()})
            for vname, (df, gt) in variants.items():
                for aname, kw in ABLATIONS.items():
                    r = reg.infer(df, use_memory=False, **kw)
                    flagged = {c for p in r.ambiguous for c in p} | {c for c, v in r.scores.items() if v["status"] != "auto"}
                    correct = sum(r.mapping.get(c) == s for c, s in gt.items())
                    wrong = [c for c, s in gt.items() if c in r.mapping and r.mapping[c] != s]
                    silent = [c for c in wrong if c not in flagged]
                    out.append({"seed": seed, "era": era, "headers": vname, "evidence": aname,
                                "n_cols": len(gt), "correct": correct, "wrong": len(wrong),
                                "silent_wrong": len(silent), "unmapped": len(gt) - correct - len(wrong),
                                "flagged": len(flagged)})
        _log(f"E1 seed {seed} done")
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- helpers
def _make_guard(schema, reference, seed):
    reg = SchemaRegistry(schema).fit_reference(reference)
    return DriftGuard(reg, schema, zones=ZoneResolver(zone_centroids()), lineage_digests=False,
                      run_id=f"seed{seed}")


def _truth_frames(seed, n_days, rows, start="2024-01-01"):
    g = TripGenerator(rows, seed=seed)
    days = pd.date_range(start, periods=n_days)
    return g, days, [g.day(d) for d in days]


# --------------------------------------------------------------------------- E2
def fault_benchmark(seeds=(0, 1, 2, 3, 4), rows=6000, warmup=14, per_fault=2) -> tuple[pd.DataFrame, pd.DataFrame]:
    schema = load_schema()
    canon2raw = ERAS[MODERN_ERA]["names"]
    batches = []
    for seed in seeds:
        n_slots = len(FAULTS) * per_fault
        n_post = 2 * n_slots + 8
        g, days, truths = _truth_frames(seed, warmup + n_post, rows)
        dg = _make_guard(schema, g.day("2023-12-01"), seed)
        rng = np.random.default_rng(seed)
        schedule = {}
        order = rng.permutation([f for f in FAULTS for _ in range(per_fault)])
        for k, f in enumerate(order):
            schedule[warmup + 2 * k] = f      # faults never on consecutive days
        for i, d in enumerate(days):
            raw = to_raw(truths[i])
            if i < warmup:
                dg.process(raw, f"s{seed}-d{i}", d, warmup=True)
                if i == warmup - 1:
                    dg.end_warmup()
                continue
            f = schedule.get(i, "clean")
            lab = {"fault": "clean"}
            if f != "clean":
                raw, lab = inject(f, raw, rng, previous=to_raw(truths[i - 1]))
            rep = dg.process(raw, f"s{seed}-d{i}", d)
            checks = rep.checks()
            sev = {x["check"]: x["severity"] for x in rep.findings}
            row = {"seed": seed, "day": i, "fault": f, "status": rep.status,
                   "checks": ",".join(sorted(checks)),
                   "alarm": any(s in ("warn", "hold", "reject") for s in sev.values()) or rep.status in ("HELD", "REJECTED"),
                   "repairs": ",".join(x["check"] for x in rep.repairs)}
            if f != "clean":
                exp = EXPECTED[f]
                row["detected"] = bool(exp["detect"] & checks) if exp["detect"] else None
                row["cause_ok"] = (exp["cause"] in checks) if exp["cause"] else None
            if f in ("obfuscated_headers", "header_synonyms"):
                if f == "obfuscated_headers":
                    gt = {c: lab["truth"][r] for c, r in canon2raw.items()}
                else:
                    from .faults import SYNONYMS
                    gt = {c: SYNONYMS.get(r, r) for c, r in canon2raw.items()}
                mp = rep.mapping["mapping"]
                row["map_correct"] = sum(mp.get(c) == s for c, s in gt.items()) / len(gt)
                row["map_silent_wrong"] = (rep.status not in ("HELD",)) and any(
                    c in mp and mp[c] != s for c, s in gt.items())
            if rep.repairs and rep.status == "ACCEPTED_REPAIRED":
                loaded = dg.wh.read("trips")
                loaded = loaded[loaded["batch_id"] == rep.batch_id]
                tr = truths[i]
                cols = {"unit_change": ["trip_distance_mi", "fare_amount", "total_amount"],
                        "clock_shift": ["pickup_ts"], "column_swap": ["tip_amount", "tolls_amount"],
                        "partial_duplicates": ["fare_amount"]}
                ks = []
                for rp in rep.repairs:
                    for c in cols.get(rp["check"], []):
                        a = loaded[c]
                        b = tr[c]
                        if c == "pickup_ts":
                            a, b = a.dt.hour + a.dt.minute / 60, b.dt.hour + b.dt.minute / 60
                        ks.append(ks_2samp(a.dropna(), b.dropna()).statistic)
                row["repair_ks_vs_truth"] = float(max(ks)) if ks else None
                row["rows_loaded_vs_truth"] = len(loaded) / len(tr)
            batches.append(row)
        _log(f"E2 seed {seed} done")
    df = pd.DataFrame(batches)
    faulted = df[df["fault"] != "clean"]
    summary = faulted.groupby("fault").agg(
        n=("fault", "size"),
        detected=("detected", lambda s: s.dropna().mean() if s.notna().any() else np.nan),
        cause_ok=("cause_ok", lambda s: s.dropna().mean() if s.notna().any() else np.nan),
        held_or_rejected=("status", lambda s: s.isin(["HELD", "REJECTED"]).mean()),
        auto_repaired=("status", lambda s: (s == "ACCEPTED_REPAIRED").mean()),
        map_correct=("map_correct", "mean"),
        repair_ks=("repair_ks_vs_truth", "max"),
    ).reset_index()
    return df, summary


# --------------------------------------------------------------------------- E3
def downstream_benchmark(seeds=(0, 1, 2), rates=(0.0, 0.1, 0.25, 0.4), rows=5000, warmup=14,
                         n_days=91, test_len=21) -> pd.DataFrame:
    schema = load_schema()
    out = []
    zones = ZoneResolver(zone_centroids())
    for seed in seeds:
        g, days, truths = _truth_frames(seed, n_days, rows)
        truth_all = pd.concat(truths, ignore_index=True)
        train = (warmup + 7, n_days - test_len)
        test = (n_days - test_len, n_days)
        oracle = evaluate(truth_all, truth_all, days, N_ZONES, train, test, seed=seed)
        out.append({"seed": seed, "rate": None, "pipeline": "oracle (true data)", **oracle})
        for rate in rates:
            rng = np.random.default_rng(1000 * seed + int(rate * 100))
            dg = _make_guard(schema, g.day("2023-12-01"), seed)
            nv = NaivePipeline(schema, zones)
            faults_used = []
            for i, d in enumerate(days):
                raw = to_raw(truths[i])
                if i < warmup:
                    dg.process(raw, f"d{i}", d, warmup=True)
                    nv.process(raw, f"d{i}", d)
                    if i == warmup - 1:
                        dg.end_warmup()
                    continue
                if rng.random() < rate:
                    f = rng.choice(FAULTS)
                    raw, _ = inject(f, raw, rng, previous=to_raw(truths[i - 1]))
                    faults_used.append(str(f))
                dg.process(raw, f"d{i}", d)
                nv.process(raw, f"d{i}", d)
            unknown = {pd.Timestamp(r.batch_date) for r in dg.reports if r.status in ("HELD", "REJECTED")}
            res_g = evaluate(dg.wh.read("trips"), truth_all, days, N_ZONES, train, test, unknown, seed)
            res_n = evaluate(nv.wh.read("trips"), truth_all, days, N_ZONES, train, test, None, seed)
            nfaults = len(faults_used)
            out.append({"seed": seed, "rate": rate, "pipeline": "naive", "n_faults": nfaults, **res_n})
            out.append({"seed": seed, "rate": rate, "pipeline": "DriftGuard", "n_faults": nfaults,
                        "held_days": len(unknown), **res_g})
            _log(f"E3 seed {seed} rate {rate}: naive {res_n['wape']:.3f}  guard {res_g['wape']:.3f}  "
                 f"oracle {oracle['wape']:.3f}  ({nfaults} faults, {len(unknown)} held)")
    return pd.DataFrame(out)
