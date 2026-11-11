"""DriftGuard on REAL NYC TLC yellow-taxi data: one command, four studies.

    pip install -e ".[real]"                                # adds pyarrow
    python scripts/run_real_study.py --download             # full study
    python scripts/run_real_study.py --download --quick     # ~15 min smoke run (2 months, sampled)
    python scripts/run_real_study.py --selftest             # no internet: synthetic files in the real layout

Studies
  A  Era mapping     one month per historical layout (2009 -> 2021) mapped onto the canonical
                     schema with real headers, then again with headers scrambled; agreement
                     between the two = how well values alone identify real columns.
  B  Real quality    the study months streamed as daily batches through DriftGuard, unmodified:
                     what real rows violate, what the sentinel flags on real days (holidays,
                     weather, fare rules), and how often it holds a real batch.
  C  Faults on real  the same stream with the 12 labelled faults injected every few days:
                     detection / correct-cause rate on top of real noise, false alarms on
                     real clean days.
  D  Downstream      demand forecast (265 zones, hourly) trained on the naive pipeline's
                     warehouse vs DriftGuard's (both fed the faulted stream) vs the clean run.

Output: results_real/ (small CSV/JSON + report_real.html) and results_real.zip to share.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driftguard.faults import EXPECTED, FAULTS, inject  # noqa: E402
from driftguard.forecast import evaluate  # noqa: E402
from driftguard.pipeline import DriftGuard, NaivePipeline  # noqa: E402
from driftguard.registry import SchemaRegistry  # noqa: E402
from driftguard.transform import coerce_to_canonical, load_schema  # noqa: E402
from driftguard.warehouse import AggWarehouse  # noqa: E402

BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{}.parquet"
N_ZONES = 265
STUDY_MONTHS = ["2023-03", "2023-04", "2023-05", "2023-06"]
ERA_MONTHS = ["2009-06", "2011-06", "2014-06", "2016-03", "2018-06", "2021-06"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- io
def path_for(data: Path, month: str) -> Path | None:
    for ext in (".parquet", ".csv"):
        p = data / f"yellow_tripdata_{month}{ext}"
        if p.exists():
            return p
    return None


def download(data: Path, months: list[str]):
    data.mkdir(parents=True, exist_ok=True)
    for m in months:
        dest = data / f"yellow_tripdata_{m}.parquet"
        if dest.exists():
            continue
        log(f"downloading {m} ...")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(BASE.format(m), tmp)
        tmp.rename(dest)
        log(f"  {dest.stat().st_size / 1e6:.0f} MB")


def _fix_types(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if "datetime" in c.lower() and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        elif pd.api.types.is_datetime64_any_dtype(df[c]) and getattr(df[c].dt, "tz", None) is not None:
            df[c] = df[c].dt.tz_localize(None)
    return df


def read_month(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, low_memory=False)
    return _fix_types(df)


def read_sample(p: Path, n: int, seed: int) -> pd.DataFrame:
    """A sample without loading 14M-row historical months into memory."""
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(p)
        df = pf.read_row_group(0).to_pandas()
    else:
        df = pd.read_csv(p, nrows=max(n * 3, 50000), low_memory=False)
    df = df.sample(min(len(df), n), random_state=seed).reset_index(drop=True)
    return df  # types left raw on purpose: the registry must cope with what files contain


def find_col(df: pd.DataFrame, *needles: str) -> str:
    for c in df.columns:
        if all(n in c.lower() for n in needles):
            return c
    raise KeyError(needles)


def daily_batches(paths: list[Path], max_rows: int, seed: int, stray_log: dict):
    rng = np.random.default_rng(seed)
    for p in paths:
        month = pd.Period(re.search(r"(\d{4}-\d{2})", p.stem).group(1), "M")
        df = read_month(p)
        ts = df[find_col(df, "pickup", "datetime")]
        day = ts.dt.normalize()
        inside = day.dt.to_period("M") == month
        stray_log[str(month)] = {"rows": int(len(df)), "outside_month": int((~inside).sum()),
                                 "null_pickup": int(ts.isna().sum())}
        df, day = df[inside], day[inside]
        for d, idx in df.groupby(day).groups.items():
            part = df.loc[idx]
            if max_rows and len(part) > max_rows:
                part = part.sample(max_rows, random_state=int(rng.integers(1 << 31)))
            yield pd.Timestamp(d), part.reset_index(drop=True)
        del df


# --------------------------------------------------------------------------- A
def study_eras(data: Path, schema: dict, reference_raw: pd.DataFrame, months: list[str], seed: int):
    reg = SchemaRegistry(schema)
    m0 = reg.infer(reference_raw, use_stats=False)
    reg.fit_reference(coerce_to_canonical(reference_raw, m0.mapping, schema)[0])
    rows, summary = [], []
    rng = np.random.default_rng(seed)
    for m in months:
        p = path_for(data, m)
        if p is None:
            log(f"A: {m} not found, skipped")
            continue
        raw = read_sample(p, 100_000, seed)
        named = reg.infer(raw, use_memory=False)
        for c, v in named.scores.items():
            rows.append({"month": m, "headers": "real", "canonical": c, **v})
        cols = list(raw.columns)
        perm = rng.permutation(len(cols))
        ob = raw.iloc[:, perm].copy()
        ren = {cols[j]: f"col_{k:02d}" for k, j in enumerate(perm)}
        ob.columns = [ren[c] for c in ob.columns]
        blind = reg.infer(ob, use_memory=False)
        flagged = {c for pr in blind.ambiguous for c in pr} | {c for c, v in blind.scores.items() if v["status"] != "auto"}
        inv = {v: k for k, v in ren.items()}
        agree = disagree = silent = 0
        for c, s in named.mapping.items():
            b = blind.mapping.get(c)
            if b is None:
                continue
            if inv[b] == s:
                agree += 1
            else:
                disagree += 1
                silent += c not in flagged
        for c, v in blind.scores.items():
            rows.append({"month": m, "headers": "scrambled", "canonical": c, **v, "source": inv[v["source"]]})
        summary.append({
            "month": m, "source_columns": len(cols), "raw_headers": "|".join(cols),
            "mapped_real_headers": len(named.mapping), "unmapped_real_headers": "|".join(named.unmapped_sources),
            "missing_required": "|".join(named.missing_required), "needs_review": named.needs_review,
            "scrambled_agree": agree, "scrambled_disagree": disagree, "scrambled_silent_disagree": silent,
            "scrambled_flagged": len(flagged), "scrambled_resolved_by_invariants": len(blind.resolved_by_invariant)})
        log(f"A: {m}: {len(named.mapping)}/{len(cols)} mapped by name+stats; "
            f"scrambled agrees on {agree}/{agree + disagree}, silent disagreements {silent}")
    return pd.DataFrame(rows), pd.DataFrame(summary)


# --------------------------------------------------------------------------- B, C, D
def batch_row(rep, fault="clean", lab=None, clean_mapping=None):
    contracts = next((json.loads(s["notes"]) for s in rep.lineage if s["step"] == "contracts" and s["notes"]), {})
    r = {"date": rep.batch_date, "fault": fault, "status": rep.status, "rows_in": rep.rows_in,
         "rows_loaded": rep.rows_loaded, "rows_quarantined": rep.rows_quarantined,
         "checks": ",".join(sorted(rep.checks())),
         "alarm": any(f["severity"] in ("warn", "hold", "reject") for f in rep.findings)
         or rep.status in ("HELD", "REJECTED"),
         "causes": " | ".join(f"{f['check']}: {f['cause']}" for f in rep.findings),
         "repairs": json.dumps(rep.repairs), "ms": round(sum(s["ms"] for s in rep.lineage)),
         "violations": json.dumps(contracts.get("violations", {}))}
    if fault != "clean":
        exp = EXPECTED[fault]
        checks = rep.checks()
        r["detected"] = bool(exp["detect"] & checks) if exp["detect"] else None
        r["cause_ok"] = (exp["cause"] in checks) if exp["cause"] else None
        if fault == "obfuscated_headers" and clean_mapping:
            mp = rep.mapping["mapping"]
            gt = {c: lab["truth"][s] for c, s in clean_mapping.items() if s in lab["truth"]}
            r["map_correct"] = sum(mp.get(c) == s for c, s in gt.items()) / max(len(gt), 1)
    return r


def study_stream(paths, schema, args):
    """One pass, three pipelines in lockstep: clean DriftGuard (B), faulted DriftGuard (C),
    faulted naive (D). Keeps memory flat: one day of raw data at a time."""
    clean_wh, fault_wh, naive_wh = AggWarehouse(), AggWarehouse(), AggWarehouse()
    g_clean = DriftGuard(SchemaRegistry(schema), schema, warehouse=clean_wh, lineage_digests=False, run_id="clean")
    g_fault = DriftGuard(SchemaRegistry(schema), schema, warehouse=fault_wh, lineage_digests=False, run_id="fault")
    naive = NaivePipeline(schema, warehouse=naive_wh)
    rng = np.random.default_rng(args.seed)
    cycle = list(rng.permutation(FAULTS))
    stray: dict = {}
    rowsB, rowsC, days = [], [], []
    prev = None
    for i, (d, raw) in enumerate(daily_batches(paths, args.max_rows, args.seed, stray)):
        days.append(d)
        warm = i < args.warmup
        if i == 0:   # the first day is trusted: fit the registry reference from it, by name
            for g in (g_clean, g_fault):
                m0 = g.registry.infer(raw, use_stats=False)
                g.registry.fit_reference(coerce_to_canonical(raw, m0.mapping, schema)[0])
        rb = g_clean.process(raw, f"{d.date()}", d, warmup=warm)
        fault, lab = "clean", None
        fraw = raw
        if not warm and (i - args.warmup) % args.fault_every == args.fault_every - 1 and prev is not None:
            n_done = sum(r["fault"] != "clean" for r in rowsC)
            fault = str(cycle[n_done % len(cycle)])
            fraw, lab = inject(fault, raw, rng, previous=prev)
        rc = g_fault.process(fraw, f"{d.date()}", d, warmup=warm)
        naive.process(fraw, f"{d.date()}", d)
        if i == args.warmup - 1:
            g_clean.end_warmup()
            g_fault.end_warmup()
            log(f"warm-up done ({args.warmup} days); thresholds calibrated on real data")
        if not warm:
            rowsB.append(batch_row(rb))
            rowsC.append(batch_row(rc, fault, lab, rb.mapping["mapping"]))
            if (i - args.warmup) % 7 == 0:
                log(f"{d.date()}  clean:{rb.status:<17} {rowsB[-1]['checks'] or '-':<40} "
                    f"faulted[{fault}]:{rc.status}")
        prev = raw
    return {"B": pd.DataFrame(rowsB), "C": pd.DataFrame(rowsC), "days": pd.DatetimeIndex(days),
            "clean": g_clean, "fault": g_fault, "naive": naive, "stray": stray,
            "wh": (clean_wh, fault_wh, naive_wh)}


def study_downstream(res, args):
    days = res["days"]
    clean_wh, fault_wh, naive_wh = res["wh"]
    n = len(days)
    test_len = min(21, max(7, n // 5))
    train, test = (args.warmup + 7, n - test_len), (n - test_len, n)
    if train[1] - train[0] < 7:
        log("D: not enough days for the forecast study; skipped")
        return pd.DataFrame()
    truth = clean_wh.read("trips")
    held = {pd.Timestamp(r.batch_date) for r in res["fault"].reports if r.status in ("HELD", "REJECTED")}
    out = []
    for name, wh, unknown in (("oracle (clean run)", clean_wh, None), ("DriftGuard", fault_wh, held),
                              ("naive", naive_wh, None)):
        r = evaluate(wh.read("trips"), truth, days, N_ZONES, train, test, unknown, args.seed)
        out.append({"pipeline": name, **r, "held_days": len(held) if name == "DriftGuard" else None})
        log(f"D: {name:<20} WAPE {r['wape']:.2%}")
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- report
def write_report(out: Path, A, B, C, D, meta):
    from driftguard.report import CSS, _pct, _table, hbar
    parts = [f"<h1>DriftGuard on real NYC TLC data</h1><p class='sub'>Months {', '.join(meta['study_months'])} "
             f"as daily batches ({meta['batches']} after a {meta['warmup']}-day warm-up"
             f"{'; ≤' + format(meta['max_rows'], ',') + ' rows/day sampled' if meta['max_rows'] else ''}).</p>"]
    k = []
    if len(B):
        k += [(f"{(B.status == 'ACCEPTED').mean():.0%}", "real days accepted untouched"),
              (f"{B.alarm.mean():.0%}", "real days with any alarm (see causes below)"),
              (f"{B.rows_quarantined.sum() / max(B.rows_in.sum(), 1):.1%}", "real rows quarantined by contracts")]
    if len(C):
        fc = C[C.fault != "clean"].dropna(subset=["detected"])
        cc = C[C.fault == "clean"]
        k += [(f"{fc.detected.mean():.0%}", f"injected faults detected ({len(fc)})"),
              (f"{fc.cause_ok.mean():.0%}", "with correct cause"),
              (f"{cc.alarm.mean():.0%}", f"alarms on real clean days ({len(cc)})")]
    if len(D):
        w = D.set_index("pipeline")["wape"]
        k += [(f"{w['naive']:.1%} → {w['DriftGuard']:.1%}", f"forecast WAPE naive → DriftGuard (clean run {w.iloc[0]:.1%})")]
    parts.append("<div class='kpis'>" + "".join(f"<div class='kpi'><b>{a}</b><span>{b}</span></div>" for a, b in k) + "</div>")
    if len(A):
        parts.append("<h2>A · Historical layouts</h2><div class='card'>" + _table(A.drop(columns=["raw_headers"])) + "</div>")
    if len(B):
        viol: dict = {}
        for v in B.violations:
            for r, n in json.loads(v).items():
                viol[r] = viol.get(r, 0) + n
        vt = pd.DataFrame(sorted(viol.items(), key=lambda kv: -kv[1]), columns=["rule", "rows violating"])
        vt["share of rows"] = vt["rows violating"] / B.rows_in.sum()
        parts.append("<h2>B · What real data violates</h2><div class='card'>"
                     + hbar(list(vt.rule), list(vt["share of rows"]), maxv=max(vt["share of rows"].max(), 1e-3))
                     + _table(vt, {"share of rows": lambda v: f"{v:.2%}"}) + "</div>")
        al = B[B.alarm][["date", "status", "checks", "causes"]]
        parts.append(f"<h2>B · Real days that raised alarms ({len(al)})</h2><p class='note'>Each needs a human "
                     "verdict: real-world event (holiday, storm, rule change) or data problem?</p><div class='card'>"
                     + _table(al) + "</div>")
    if len(C):
        fc = C[C.fault != "clean"]
        s = fc.groupby("fault").agg(n=("fault", "size"), detected=("detected", "mean"), cause_ok=("cause_ok", "mean"),
                                    status=("status", lambda x: ", ".join(sorted(set(x))))).reset_index()
        parts.append("<h2>C · Injected faults on real batches</h2><div class='card'>"
                     + _table(s, {"detected": _pct, "cause_ok": _pct}) + "</div>")
    if len(D):
        parts.append("<h2>D · Downstream forecast</h2><div class='card'>"
                     + _table(D, {"wape": lambda v: f"{v:.2%}"}) + "</div>")
    html = (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>DriftGuard · real data</title>"
            f"<style>{CSS}</style></head><body><div class='viz-root'><div class='wrap'>{''.join(parts)}</div></div></body></html>")
    (out / "report_real.html").write_text(html)


# --------------------------------------------------------------------------- selftest data
def make_selftest_data(root: Path, study_months: list[str], era_months: list[str], rows: int):
    from driftguard.synth import TripGenerator, to_raw
    g = TripGenerator(rows, seed=42)
    for m in study_months:
        per = pd.Period(m, "M")
        days = pd.date_range(per.start_time, per.end_time.normalize())
        df = pd.concat([to_raw(g.day(d), "2023") for d in days], ignore_index=True)
        df = df.rename(columns={"Airport_fee": "airport_fee"})        # as in real 2023 files
        df.loc[df.sample(frac=0.003, random_state=1).index, "payment_type"] = 0   # stable oddity
        df.to_csv(root / f"yellow_tripdata_{m}.csv", index=False)
    era_of = {"2009": "2009", "2011": "2010", "2014": "2010", "2016": "2015", "2018": "2017", "2021": "2019"}
    for m in era_months:
        era = era_of.get(m[:4], "2019")
        pd.concat([to_raw(g.day(f"{m}-0{k}"), era) for k in range(1, 4)]).to_csv(
            root / f"yellow_tripdata_{m}.csv", index=False)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="results_real")
    ap.add_argument("--months", nargs="+", default=STUDY_MONTHS)
    ap.add_argument("--era-months", nargs="*", default=ERA_MONTHS)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--quick", action="store_true", help="2 months, 30k rows/day, 3 era months")
    ap.add_argument("--selftest", action="store_true", help="synthetic files in the real layout, no internet")
    ap.add_argument("--max-rows", type=int, default=0, help="sample at most N rows per day (0 = all)")
    ap.add_argument("--warmup", type=int, default=14)
    ap.add_argument("--fault-every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.quick:
        args.months, args.era_months = args.months[:2], args.era_months[:: 2]
        args.max_rows = args.max_rows or 30_000
    tmp = None
    if args.selftest:
        tmp = Path(tempfile.mkdtemp(prefix="dg_selftest_"))
        args.months, args.era_months = ["2023-03", "2023-04"], ["2009-06", "2014-06", "2018-06"]
        make_selftest_data(tmp, args.months, args.era_months, rows=3000)
        args.data = str(tmp)
        log(f"selftest data written to {tmp}")
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.download:
        download(data, args.months + args.era_months)

    schema = load_schema()
    paths = [path_for(data, m) for m in args.months]
    missing = [m for m, p in zip(args.months, paths) if p is None]
    if missing:
        sys.exit(f"missing study months {missing} in {data} (use --download)")
    t0 = time.time()

    log("Study A: historical layouts")
    A_rows, A = study_eras(data, schema, read_sample(paths[0], 100_000, args.seed), args.era_months, args.seed)
    A_rows.to_csv(out / "A_mapping_rows.csv", index=False)
    A.to_csv(out / "A_eras.csv", index=False)

    log("Studies B + C: streaming daily batches (clean and faulted, in lockstep)")
    res = study_stream(paths, schema, args)
    res["B"].to_csv(out / "B_real_batches.csv", index=False)
    res["C"].to_csv(out / "C_faulted_batches.csv", index=False)
    res["wh"][0].read("quarantine_reasons").to_csv(out / "B_quarantine_reasons.csv", index=False)
    res["wh"][0].read("findings").to_csv(out / "B_findings.csv", index=False)
    (out / "B_calibrated_thresholds.json").write_text(json.dumps(res["clean"].sentinel.th, indent=2))

    log("Study D: downstream forecast")
    D = study_downstream(res, args)
    D.to_csv(out / "D_downstream.csv", index=False)

    B, C = res["B"], res["C"]
    fc = C[C.fault != "clean"].dropna(subset=["detected"]) if len(C) else C
    meta = {"selftest": bool(args.selftest), "study_months": args.months, "era_months": args.era_months,
            "max_rows": args.max_rows, "warmup": args.warmup, "fault_every": args.fault_every,
            "batches": int(len(B)), "minutes": round((time.time() - t0) / 60, 1),
            "month_files": res["stray"],
            "B_days_accepted_untouched": float((B.status == "ACCEPTED").mean()) if len(B) else None,
            "B_days_with_alarm": float(B.alarm.mean()) if len(B) else None,
            "B_rows_quarantined_share": float(B.rows_quarantined.sum() / max(B.rows_in.sum(), 1)) if len(B) else None,
            "C_faults": int(len(fc)), "C_detected": float(fc.detected.mean()) if len(fc) else None,
            "C_cause_ok": float(fc.cause_ok.mean()) if len(fc) else None,
            "C_clean_day_alarm_rate": float(C[C.fault == "clean"].alarm.mean()) if len(C) else None,
            "D": D.to_dict("records")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    write_report(out, A, B, C, D, meta)
    z = shutil.make_archive(str(out), "zip", out)
    log(f"done in {meta['minutes']} min -> {out}/report_real.html ; share {z}")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("D", "month_files")}, indent=2, default=str))
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
